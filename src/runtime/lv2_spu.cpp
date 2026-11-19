// LV2 SPU thread groups: sys_spu_image_*, sys_spu_thread_group_*, sys_spu_thread_*,
// and the SPU side of the system (channels, mailboxes, signal notification, event
// ports / queues, MFC DMA and lock-line atomics). Each SPU thread runs the SPU
// interpreter on its own host thread.
//
// Structure layouts, syscall numbers and the stop/mailbox protocol of the SPU
// runtime library follow public PS3 SDK documentation.

#define _CRT_SECURE_NO_WARNINGS
#include "lv2_spu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hle.h"
#include "profiler.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace lv2 {
namespace {

void spu_idle_wait() {
#if defined(_WIN32)
  thread_local HANDLE timer = [] {
    HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /*HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
    return h ? h : CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
  }();
  LARGE_INTEGER due;
  due.QuadPart = -10;  // ~0.5 ms on Windows high-res timer
  if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
    WaitForSingleObject(timer, INFINITE);
    return;
  }
#endif
  std::this_thread::sleep_for(std::chrono::microseconds(250));
}

Kernel& K() { return Kernel::get(); }
uint8_t* G(uint32_t a) { return K().base() + a; }

// PS3RECOMP_SPU_TRACE=1: log the first channel/DMA/stop events of every SPU thread.
bool spu_trace() {
  static const bool on = std::getenv("PS3RECOMP_SPU_TRACE") != nullptr;
  return on;
}
void trace(SpuThread& t, const char* fmt, ...) {
  if (!spu_trace()) return;
  thread_local int count = 0;
  thread_local std::string recent[4];
  thread_local int recent_i = 0;
  if (count > 2000) return;
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  char line[300];
  std::snprintf(line, sizeof line, "pc=0x%05X %s", t.core.pc, buf);
  for (auto& r : recent) if (r == line) return;  // polling loops repeat the same few events
  recent[recent_i++ & 3] = line;
  ++count;
  K().logf("spu[%s #%u] %s", t.name.c_str(), t.index, line);
}

// Waits on t.cv for `p`, waking periodically so group termination is noticed.
template <class Pred> void wait_on(SpuThread& t, std::unique_lock<std::mutex>& lk, Pred p) {
  while (!p()) {
    t.check_stop();
    t.cv.wait_for(lk, std::chrono::milliseconds(5));
  }
}

// Effective address -> host pointer, honouring the SPU thread LS window.
uint8_t* ea_ptr(SpuThread& t, uint32_t ea, uint32_t len) {
  // The first 64 KB of the PS3 address space is never mapped (executables start at
  // 0x10000): DMA there is a guest bug (e.g. a null pointer), reported by the caller.
  if (ea < 0x10000) return nullptr;
  if (ea >= kSpuThreadBase) {
    const uint32_t idx = (ea - kSpuThreadBase) / kSpuThreadStride, off = ea % kSpuThreadStride;
    auto g = t.group.lock();
    if (g && idx < g->threads.size() && g->threads[idx] && off + len <= spu::kLsSize)
      return g->threads[idx]->core.ls() + off;
    return nullptr;
  }
  return G(ea);
}

void deliver_snr(SpuThread& dst, int n, uint32_t value) {
  std::lock_guard<std::mutex> lk(dst.m);
  dst.snr[n] = dst.snr_or[n] && dst.snr_valid[n] ? dst.snr[n] | value : value;
  dst.snr_valid[n] = true;
  dst.event_stat |= n == 0 ? kEventS1 : kEventS2;
  dst.cv.notify_all();
}

// Writes to another SPU thread's signal-notification registers via the EA window.
bool ea_signal(SpuThread& t, uint32_t ea, uint32_t value) {
  if (ea < kSpuThreadBase) return false;
  const uint32_t idx = (ea - kSpuThreadBase) / kSpuThreadStride, off = ea % kSpuThreadStride;
  if (off != 0x5400C && off != 0x5C00C) return false;
  auto g = t.group.lock();
  if (!g || idx >= g->threads.size() || !g->threads[idx]) return false;
  deliver_snr(*g->threads[idx], off == 0x5400C ? 0 : 1, value);
  return true;
}

}  // namespace

std::mutex& spu_line_mutex() {
  static std::mutex m;
  return m;
}

void SpuThread::push_in(uint32_t v) {
  std::lock_guard<std::mutex> lk(m);
  in_mbox.push_back(v);
  cv.notify_all();
}

uint32_t SpuThread::pop_out() {
  std::lock_guard<std::mutex> lk(m);
  if (out_mbox.empty()) return 0;
  const uint32_t v = out_mbox.front();
  out_mbox.pop_front();
  return v;
}

bool SpuThread::reservation_lost() {
  if (!res_valid) return false;
  const uint8_t* p = ea_ptr(*this, res_ea, 128);
  if (p && std::memcmp(p, res_data, 128) == 0) return false;
  res_valid = false;
  event_stat |= kEventLR;
  return true;
}

void SpuThread::mfc_cmd(uint32_t cmd) {
  const uint32_t ea = eal;  // 32-bit effective addresses
  trace(*this, "MFC cmd 0x%02X lsa 0x%05X ea 0x%08X size 0x%X tag %u", cmd, lsa, eal, size, tag);
  uint8_t* ls = core.ls();
  const uint32_t l = lsa & spu::kLsMask;
  const auto transfer = [&](bool put, uint32_t lsaddr, uint32_t eaddr, uint32_t len) {
    if (!len) return;
    uint8_t* p = ea_ptr(*this, eaddr, len);
    if (!p) {
      static std::atomic<int> reported{0};
      if (reported++ < 20)
        K().logf("SPU %s #%u: DMA %s unmapped EA 0x%08X (%u bytes, cmd 0x%02X, lsa 0x%05X, pc 0x%05X)", name.c_str(), index,
                 put ? "to" : "from", eaddr, len, cmd, lsaddr, core.pc);
      return;
    }
    lsaddr &= spu::kLsMask;
    len = std::min(len, spu::kLsSize - lsaddr);
    if (put) {
      // Debug: PS3RECOMP_WATCH=<hex address> also reports SPU DMA writes covering it.
      static const uint32_t watch = [] {
        const char* w = std::getenv("PS3RECOMP_WATCH");
        return w ? uint32_t(std::strtoul(w, nullptr, 16)) : 0u;
      }();
      if (watch && eaddr <= watch && watch < eaddr + len) {
        static std::atomic<int> reported{0};
        if (reported++ < 20)
          K().logf("watch: SPU %s #%u (pc 0x%05X) PUT %u bytes to 0x%08X covering 0x%08X", name.c_str(), index, core.pc, len,
                   eaddr, watch);
      }
      std::memcpy(p, ls + lsaddr, len);
      same_get = 0;
    } else {
      bool cached = false;
      for (int i = 0; i < 4; ++i)
        if (get_cache_lsa[i] == lsaddr && get_cache_ea[i] == eaddr && get_cache_len[i] == len) {
          cached = true;
          break;
        }
      if (cached && std::memcmp(ls + lsaddr, p, len) == 0) {
        if (++same_get > 32) {
          check_stop();
          spu_idle_wait();
        } else if (same_get > 8) {
          std::this_thread::yield();
        }
      } else {
        std::memcpy(ls + lsaddr, p, len);
        core.invalidate_ls(lsaddr, len);
        same_get = 0;
        if (!cached) {
          const uint32_t slot = get_cache_next++ & 3;
          get_cache_lsa[slot] = lsaddr;
          get_cache_ea[slot] = eaddr;
          get_cache_len[slot] = len;
        }
      }
    }
  };
  switch (cmd) {
    case 0x20: case 0x21: case 0x22: case 0x30: case 0x31: case 0x32:  // PUT[R][B|F]
      transfer(true, l, ea, size);
      break;
    case 0x40: case 0x41: case 0x42:  // GET[B|F]
      transfer(false, l, ea, size);
      break;
    case 0x24: case 0x25: case 0x26: case 0x34: case 0x35: case 0x36:  // PUTL[R]
    case 0x44: case 0x45: case 0x46: {                                   // GETL
      const bool put = (cmd & 0xF0) != 0x40;
      uint32_t dst = l;
      for (uint32_t off = 0; off + 8 <= size; off += 8) {
        const uint32_t w0 = core.ls32(eal + off), w1 = core.ls32(eal + off + 4);
        const uint32_t n = w0 & 0x7FFF;
        // Cell MFC: for list elements smaller than 16 bytes, LS offset within the
        // 16-byte quadword matches the low 4 bits of EA.
        const uint32_t ls_a = (n > 0 && n < 16) ? ((dst & ~15u) | (w1 & 15u)) : dst;
        transfer(put, ls_a, w1, n);
        dst += (n + 15) & ~15u;
        // Stall-and-notify (bit 31) is not modelled: the list runs to completion.
      }
      break;
    }
    case 0xD0: {  // GETLLAR
      const uint32_t line = ea & ~127u;
      uint8_t* p = ea_ptr(*this, line, 128);
      if (res_valid && res_ea == line && p && std::memcmp(p, res_data, 128) == 0) {
        std::memcpy(ls + (l & ~127u), res_data, 128);
        if (++same_getllar > 32) {
          check_stop();
          spu_idle_wait();
        } else if (same_getllar > 8) {
          std::this_thread::yield();
        }
      } else {
        std::lock_guard<std::mutex> lk(spu_line_mutex());
        if (p) {
          std::memcpy(res_data, p, 128);
          std::memcpy(ls + (l & ~127u), res_data, 128);
          core.invalidate_ls(l & ~127u, 128);
        }
        same_getllar = 0;
      }
      res_ea = line;
      res_valid = p != nullptr;
      atomic_stat = 4;  // MFC_GETLLAR_SUCCESS
      atomic_avail = true;
      break;
    }
    case 0xB4: {  // PUTLLC
      // The PPU writes guest memory without taking our lock, so the line is
      // committed as compare-and-swaps of only the 8-byte chunks the SPU changed:
      // concurrent PPU stores to other words of the line are never overwritten
      // with stale data, and a PPU store to a changed chunk makes PUTLLC fail.
      std::lock_guard<std::mutex> lk(spu_line_mutex());
      const uint32_t line = ea & ~127u;
      uint8_t* p = ea_ptr(*this, line, 128);
      bool ok = res_valid && res_ea == line && p && std::memcmp(p, res_data, 128) == 0;
      if (ok) {
        const uint8_t* src = ls + (l & ~127u);
        int done[16];
        int n = 0;
        for (int i = 0; i < 16 && ok; ++i) {
          uint64_t oldv, newv;
          std::memcpy(&oldv, res_data + 8 * i, 8);
          std::memcpy(&newv, src + 8 * i, 8);
          if (oldv == newv) continue;
          std::atomic_ref<uint64_t> word(*reinterpret_cast<uint64_t*>(p + 8 * i));
          uint64_t expected = oldv;
          if (word.compare_exchange_strong(expected, newv)) done[n++] = i;
          else ok = false;
        }
        if (!ok)  // undo the chunks already committed
          for (int k = 0; k < n; ++k) {
            uint64_t oldv, newv;
            std::memcpy(&oldv, res_data + 8 * done[k], 8);
            std::memcpy(&newv, src + 8 * done[k], 8);
            std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t*>(p + 8 * done[k])).compare_exchange_strong(newv, oldv);
          }
      }
      res_valid = false;
      same_getllar = 0;
      atomic_stat = ok ? 0 : 1;  // MFC_PUTLLC_SUCCESS / FAILURE
      atomic_avail = true;
      break;
    }
    case 0xB0: case 0xB8: {  // PUTLLUC, PUTQLLUC
      std::lock_guard<std::mutex> lk(spu_line_mutex());
      uint8_t* p = ea_ptr(*this, ea & ~127u, 128);
      if (p) std::memcpy(p, ls + (l & ~127u), 128);
      res_valid = false;
      same_getllar = 0;
      atomic_stat = 2;  // MFC_PUTLLUC_SUCCESS
      atomic_avail = true;
      break;
    }
    case 0xA0: case 0xA1: case 0xA2:  // SNDSIG[B|F]
      if (!ea_signal(*this, ea, core.ls32(l))) K().logf("SPU %s: SNDSIG to 0x%08X ignored", name.c_str(), ea);
      break;
    case 0x89: {  // SDCRZ: zero a cache line
      uint8_t* p = ea_ptr(*this, ea & ~127u, 128);
      if (p) std::memset(p, 0, 128);
      break;
    }
    default:
      break;  // BARRIER / EIEIO / SYNC and cache hints: DMA here is synchronous
  }
}

uint32_t SpuThread::read_channel(spu::SpuCore&, uint32_t ch) {
  switch (ch) {
    case spu::MFC_RdTagStat: return tag_mask;  // all DMA completes immediately
    case spu::MFC_RdAtomicStat: atomic_avail = false; return atomic_stat;
    case spu::MFC_RdListStallStat: return 0;
    case spu::MFC_RdTagMask: return tag_mask;
    case spu::SPU_RdMachStat: return 0;
    default: break;
  }
  trace(*this, "rdch %u", ch);
  std::unique_lock<std::mutex> lk(m);
  switch (ch) {
    case spu::SPU_RdInMbox: {
      wait_on(*this, lk, [&] { return !in_mbox.empty(); });
      const uint32_t v = in_mbox.front();
      in_mbox.pop_front();
      return v;
    }
    case spu::SPU_RdSigNotify1: case spu::SPU_RdSigNotify2: {
      const int n = ch == spu::SPU_RdSigNotify1 ? 0 : 1;
      wait_on(*this, lk, [&] { return snr_valid[n]; });
      snr_valid[n] = false;
      event_stat &= ~(n == 0 ? kEventS1 : kEventS2);
      return snr[n];
    }
    case spu::SPU_RdEventMask: return event_mask;
    case spu::SPU_RdEventStat:
      for (;;) {
        reservation_lost();
        if (event_stat & event_mask) return event_stat & event_mask;
        check_stop();
        lk.unlock();
        std::this_thread::yield();
        lk.lock();
      }
    default:
      K().logf("SPU %s: read of unhandled channel %u", name.c_str(), ch);
      return 0;
  }
}

void SpuThread::write_channel(spu::SpuCore&, uint32_t ch, uint32_t v) {
  if (ch >= spu::SPU_WrOutMbox || ch <= spu::SPU_WrEventAck) trace(*this, "wrch %u = 0x%X", ch, v);
  switch (ch) {
    case spu::MFC_LSA: lsa = v; return;
    case spu::MFC_EAH: eah = v; return;
    case spu::MFC_EAL: eal = v; return;
    case spu::MFC_Size: size = v & 0xFFFF; return;
    case spu::MFC_TagID: tag = v & 31; return;
    case spu::MFC_Cmd: mfc_cmd(v & 0xFF); return;
    case spu::MFC_WrTagMask: tag_mask = v; return;
    case spu::MFC_WrTagUpdate: case spu::MFC_WrListStallAck: case spu::MFC_WrMSSyncReq: return;
    case spu::SPU_WrEventMask: { std::lock_guard<std::mutex> lk(m); event_mask = v; return; }
    case spu::SPU_WrEventAck: { std::lock_guard<std::mutex> lk(m); event_stat &= ~v; return; }
    case spu::SPU_WrOutMbox: {
      std::lock_guard<std::mutex> lk(m);
      out_mbox.clear();  // depth 1
      out_mbox.push_back(v);
      return;
    }
    case spu::SPU_WrOutIntrMbox: {
      // sys_spu_thread_send_event / throw_event / event-flag set: code in the top byte.
      const uint32_t code = v >> 24, data0 = v & 0xFFFFFF;
      if (code < 128) {
        const uint32_t spup = code & 63;
        const uint32_t data1 = pop_out();
        const uint32_t q = ports[spup];
        uint32_t rc = CELL_ENOTCONN;
        if (q) rc = event_queue_send(q, kSpuThreadEventUserKey, id, uint64_t(spup) << 32 | data0, data1);
        if (rc == CELL_ESRCH) rc = CELL_ENOTCONN;
        if (code < 64) push_in(rc);  // send_event waits for the result; throw_event does not
      } else if (code == 128 || code == 192) {
        const uint32_t flag = pop_out();
        K().logf("SPU %s: sys_event_flag_set_bit(0x%X, %u) not supported", name.c_str(), flag, data0);
        if (code == 128) push_in(CELL_OK);
      } else {
        K().logf("SPU %s: unknown interrupt-mailbox code 0x%08X", name.c_str(), v);
      }
      return;
    }
    default:
      K().logf("SPU %s: write of unhandled channel %u = 0x%X", name.c_str(), ch, v);
  }
}

uint32_t SpuThread::channel_count(spu::SpuCore&, uint32_t ch) {
  switch (ch) {
    case spu::MFC_RdAtomicStat: return atomic_avail ? 1 : 0;
    case spu::MFC_Cmd: return 16;
    case spu::SPU_WrOutIntrMbox: return 1;
    default: break;
  }
  std::lock_guard<std::mutex> lk(m);
  switch (ch) {
    case spu::SPU_RdInMbox: return uint32_t(in_mbox.size());
    case spu::SPU_WrOutMbox: return out_mbox.empty() ? 1 : 0;
    case spu::SPU_RdSigNotify1: return snr_valid[0] ? 1 : 0;
    case spu::SPU_RdSigNotify2: return snr_valid[1] ? 1 : 0;
    case spu::SPU_RdEventStat: reservation_lost(); return (event_stat & event_mask) ? 1 : 0;
    default: return 1;
  }
}

bool SpuThread::stop(spu::SpuCore&, uint32_t code) {
  trace(*this, "stop 0x%X", code);
  if (hle_stop) {
    bool cont = false;
    if (hle_stop(*this, code, cont)) return cont;
  }
  switch (code) {
    case 0x110: case 0x111: {  // sys_spu_thread_receive_event / tryreceive_event
      const uint32_t spuq = pop_out();
      uint32_t q = 0;
      {
        std::lock_guard<std::mutex> lk(m);
        for (auto& [n, qid] : queues) if (n == spuq) q = qid;
      }
      if (!q) { push_in(CELL_EINVAL); return true; }
      uint64_t e[4];
      uint32_t rc;
      if (code == 0x110) {
        // Blocking receive that still notices group termination.
        for (;;) {
          rc = event_queue_receive(q, e, false);
          if (rc != CELL_EBUSY) break;
          check_stop();
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
      } else {
        rc = event_queue_receive(q, e, false);
      }
      if (rc != CELL_OK) { push_in(rc); return true; }
      std::lock_guard<std::mutex> lk(m);
      in_mbox.push_back(CELL_OK);
      in_mbox.push_back(uint32_t(e[1]));
      in_mbox.push_back(uint32_t(e[2]));
      in_mbox.push_back(uint32_t(e[3]));
      cv.notify_all();
      return true;
    }
    case 0x101: {  // sys_spu_thread_group_exit
      const uint32_t status = pop_out();
      if (auto g = group.lock()) spu_group_finish(*g, kJoinGroupExit, status);
      throw SpuTerminated{};
    }
    case 0x102:  // sys_spu_thread_exit
      exit_status = pop_out();
      return false;
    default:
      K().logf("SPU %s: stop 0x%X at 0x%05X", name.c_str(), code, core.pc);
      exit_status = code;
      return false;
  }
}

// Ends the group: every SPU thread is asked to stop, joiners are woken.
void spu_group_finish(SpuGroup& g, uint32_t cause, uint32_t status) {
  std::lock_guard<std::mutex> lk(g.m);
  if (g.done) return;
  g.done = true;
  g.cause = cause;
  g.status = status;
  for (auto& t : g.threads)
    if (t) { t->core.request_exit(); t->cv.notify_all(); }
  g.cv.notify_all();
}

namespace {

void load_image(SpuThread& t) {
  uint8_t* ls = t.core.ls();
  std::memset(ls, 0, spu::kLsSize);
  for (const SpuSegment& s : t.segs) {
    if (s.size <= 0 || s.ls >= spu::kLsSize) continue;
    const uint32_t n = std::min<uint32_t>(uint32_t(s.size), spu::kLsSize - s.ls);
    if (s.type == kSegCopy) std::memcpy(ls + s.ls, G(s.addr_or_value), n);
    else if (s.type == kSegFill)
      for (uint32_t i = 0; i + 4 <= n; i += 4) t.core.set_ls32(s.ls + i, s.addr_or_value);
  }
  for (int i = 0; i < 4; ++i) t.core.gpr[3 + i].set_d(0, t.args[i]);
  t.core.gpr[1].set_w(0, spu::kLsSize - 0x10);  // stack pointer: top of LS
  t.core.pc = t.entry & spu::kLsMask;
}

void run_spu_thread(std::shared_ptr<SpuGroup> g, std::shared_ptr<SpuThread> t) {
  prof::register_thread("spu:" + t->name + "#" + std::to_string(t->index));
  prof::register_guest_pc(&t->core.pc, t->core.ls(), spu::kLsSize);
  try {
    if (t->hle_main) t->hle_main(*t);
    else t->core.run();
  } catch (const SpuTerminated&) {
  } catch (const spu::SpuHalt& h) {
    K().logf("SPU %s halted at 0x%05X: %s", t->name.c_str(), h.pc, h.why);
  } catch (const std::exception& e) {
    K().logf("SPU %s: %s", t->name.c_str(), e.what());
  }
  std::lock_guard<std::mutex> lk(g->m);
  if (--g->running == 0 && !g->done) {
    g->done = true;
    g->cause = kJoinAllThreadsExit;
    g->cv.notify_all();
  }
}

}  // namespace

std::shared_ptr<SpuGroup> spu_group_create(const std::string& name, uint32_t num, int32_t prio) {
  auto g = std::make_shared<SpuGroup>();
  g->name = name;
  g->num = num;
  g->prio = prio;
  g->threads.resize(8);  // slots are indexed by spu_num (0-7), independent of `num`
  g->id = K().add_object(g);
  K().logf("SPU thread group '%s' created (%u threads, id 0x%X)", name.c_str(), num, g->id);
  return g;
}

std::shared_ptr<SpuThread> spu_thread_create(const std::shared_ptr<SpuGroup>& g, uint32_t index, const std::string& name) {
  auto t = std::make_shared<SpuThread>();
  t->group = g;
  t->index = index;
  t->name = name;
  t->id = K().add_object(t);
  std::lock_guard<std::mutex> lk(g->m);
  g->threads[index] = t;
  return t;
}

uint32_t spu_group_start(uint32_t group_id) {
  auto g = K().get_object<SpuGroup>(group_id);
  if (!g) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(g->m);
  if (g->started && !g->done) return CELL_EBUSY;
  for (auto& t : g->threads) if (t && t->host.joinable()) t->host.join();
  g->started = true;
  g->done = false;
  g->running = 0;
  g->cause = g->status = 0;
  for (auto& t : g->threads) {
    if (!t) continue;
    t->core.reset();
    {
      std::lock_guard<std::mutex> tl(t->m);
      t->in_mbox.clear(); t->out_mbox.clear();
      t->snr_valid[0] = t->snr_valid[1] = false;
      t->res_valid = false; t->event_stat = 0;
    }
    if (!t->hle_main) load_image(*t);
    ++g->running;
    t->host = std::thread(run_spu_thread, g, t);
  }
  K().logf("SPU thread group '%s' started (%u threads)", g->name.c_str(), g->running);
  return CELL_OK;
}

uint32_t spu_group_join(uint32_t group_id, uint32_t* cause, uint32_t* status) {
  auto g = K().get_object<SpuGroup>(group_id);
  if (!g) return CELL_ESRCH;
  {
    std::unique_lock<std::mutex> lk(g->m);
    if (!g->started) return CELL_ESRCH;
    g->cv.wait(lk, [&] { return g->done; });
  }
  for (auto& t : g->threads) if (t && t->host.joinable()) t->host.join();
  std::lock_guard<std::mutex> lk(g->m);
  g->started = false;
  if (cause) *cause = g->cause;
  if (status) *status = g->status;
  return CELL_OK;
}

// ---- syscalls ----------------------------------------------------------------
namespace {

#define ARG(n) (c.r[3 + (n)])
#define A32(n) uint32_t(c.r[3 + (n)])

// Reads a guest sys_spu_image {u32 type, u32 entry, u32 segs, s32 nsegs}.
bool read_image(uint32_t img, uint32_t& entry, std::vector<SpuSegment>& segs) {
  if (!img) return false;
  entry = K().r32(img + 4);
  const uint32_t sp = K().r32(img + 8);
  const int32_t n = int32_t(K().r32(img + 12));
  if (n < 0 || n > 256) return false;
  segs.clear();
  for (int32_t i = 0; i < n; ++i) {
    const uint32_t a = sp + 16 * uint32_t(i);
    segs.push_back(SpuSegment{int32_t(K().r32(a)), K().r32(a + 4), int32_t(K().r32(a + 8)), K().r32(a + 12)});
  }
  return true;
}

uint64_t group_create(ppu::Context& c) {
  // (u32* id, u32 num, s32 prio, sys_spu_thread_group_attribute* {u32 nsize, u32 name, s32 type, u32 ct})
  const uint32_t num = A32(1);
  if (num == 0 || num > 6) return CELL_EINVAL;
  std::string name;
  uint32_t type = 0;
  if (const uint32_t attr = A32(3)) {
    name = K().read_str(K().r32(attr + 4), 64);
    type = K().r32(attr + 8);
  }
  auto g = spu_group_create(name, num, int32_t(ARG(2)));
  g->type = type;
  K().w32(A32(0), g->id);
  return CELL_OK;
}

uint64_t thread_initialize(ppu::Context& c) {
  // (u32* thread, u32 group, u32 spu_num, sys_spu_image* img, attr* {u32 name, u32 nsize, u32 option}, arg* {u64 x4})
  auto g = K().get_object<SpuGroup>(A32(1));
  if (!g) { K().logf("sys_spu_thread_initialize: no group 0x%X", A32(1)); return CELL_ESRCH; }
  const uint32_t idx = A32(2);
  if (idx >= g->threads.size()) { K().logf("sys_spu_thread_initialize: spu_num %u out of range", idx); return CELL_EINVAL; }
  uint32_t entry = 0;
  std::vector<SpuSegment> segs;
  if (!read_image(A32(3), entry, segs)) {
    K().logf("sys_spu_thread_initialize: bad image at 0x%08X (nsegs %d)", A32(3), int32_t(K().r32(A32(3) + 12)));
    return CELL_EINVAL;
  }
  std::string name;
  if (const uint32_t attr = A32(4)) name = K().read_str(K().r32(attr), 64);
  auto t = spu_thread_create(g, idx, name);
  t->entry = entry;
  t->segs = std::move(segs);
  if (const uint32_t arg = A32(5))
    for (int i = 0; i < 4; ++i) t->args[i] = K().r64(arg + 8 * i);
  K().w32(A32(0), t->id);
  return CELL_OK;
}

uint64_t group_join(ppu::Context& c) {
  uint32_t cause = 0, status = 0;
  const uint32_t rc = spu_group_join(A32(0), &cause, &status);
  if (rc == CELL_OK) { K().w32(A32(1), cause); K().w32(A32(2), status); }
  return rc;
}

uint64_t group_terminate(ppu::Context& c) {
  auto g = K().get_object<SpuGroup>(A32(0));
  if (!g) return CELL_ESRCH;
  spu_group_finish(*g, kJoinTerminated, A32(1));
  return CELL_OK;
}

uint64_t group_destroy(ppu::Context& c) {
  auto g = K().get_object<SpuGroup>(A32(0));
  if (!g) return CELL_ESRCH;
  {
    std::lock_guard<std::mutex> lk(g->m);
    if (g->started && !g->done) return CELL_EBUSY;
  }
  for (auto& t : g->threads) {
    if (!t) continue;
    if (t->host.joinable()) t->host.join();
    K().remove_object(t->id);
  }
  K().remove_object(g->id);
  return CELL_OK;
}

std::shared_ptr<SpuThread> thread_arg(ppu::Context& c) { return K().get_object<SpuThread>(A32(0)); }

}  // namespace

bool spu_syscall(uint32_t n, ppu::Context& c, uint64_t& rc) {
  rc = CELL_OK;
  switch (n) {
    case 169: return true;  // sys_spu_initialize
    case 170: rc = group_create(c); return true;
    case 171: rc = group_destroy(c); return true;
    case 172: rc = thread_initialize(c); return true;
    case 173: rc = spu_group_start(A32(0)); return true;
    case 174: case 175: case 176: return true;  // suspend / resume / yield
    case 177: rc = group_terminate(c); return true;
    case 178: rc = group_join(c); return true;
    case 179: return true;  // set_priority
    case 180: if (auto g = K().get_object<SpuGroup>(A32(0))) K().w32(A32(1), uint32_t(g->prio)); else rc = CELL_ESRCH; return true;
    case 181: {  // sys_spu_thread_write_ls(thread, lsa, u64 value, type)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      const uint32_t lsa = A32(1) & spu::kLsMask, type = A32(3);
      uint8_t* p = t->core.ls() + lsa;
      const uint64_t v = ARG(2);
      for (uint32_t i = 0; i < type && lsa + i < spu::kLsSize; ++i) p[i] = uint8_t(v >> (8 * (type - 1 - i)));
      return true;
    }
    case 182: {  // sys_spu_thread_read_ls(thread, lsa, u64* value, type)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      const uint32_t lsa = A32(1) & spu::kLsMask, type = A32(3);
      const uint8_t* p = t->core.ls() + lsa;
      uint64_t v = 0;
      for (uint32_t i = 0; i < type && lsa + i < spu::kLsSize; ++i) v = v << 8 | p[i];
      K().w64(A32(2), v);
      return true;
    }
    case 184: {  // sys_spu_thread_write_snr(thread, number, value)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      deliver_snr(*t, A32(1) ? 1 : 0, A32(2));
      return true;
    }
    case 185: {  // sys_spu_thread_group_connect_event(group, eq, et)
      auto g = K().get_object<SpuGroup>(A32(0));
      if (!g) { rc = CELL_ESRCH; return true; }
      if (A32(2) == 1) g->run_queue = A32(1);  // SYS_SPU_THREAD_GROUP_EVENT_RUN
      return true;
    }
    case 186: return true;
    case 187: {  // sys_spu_thread_set_spu_cfg(thread, u64 value): bit0/1 = SNR1/2 OR mode
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      t->snr_or[0] = ARG(1) & 1;
      t->snr_or[1] = (ARG(1) & 2) != 0;
      return true;
    }
    case 188: {
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      K().w64(A32(1), (t->snr_or[0] ? 1 : 0) | (t->snr_or[1] ? 2 : 0));
      return true;
    }
    case 190: {  // sys_spu_thread_write_spu_mb(thread, value)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      t->push_in(A32(1));
      return true;
    }
    case 191: {  // sys_spu_thread_connect_event(thread, eq, et, spup)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      const uint32_t spup = A32(3);
      if (A32(2) != 1 || spup > 63) { rc = CELL_EINVAL; return true; }  // SYS_SPU_THREAD_EVENT_USER only
      if (t->ports[spup]) { rc = CELL_EISCONN; return true; }
      t->ports[spup] = A32(1);
      return true;
    }
    case 192: {  // disconnect_event(thread, et, spup)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      if (A32(2) < 64) t->ports[A32(2)] = 0;
      return true;
    }
    case 193: {  // sys_spu_thread_bind_queue(thread, eq, spuq_num)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      std::lock_guard<std::mutex> lk(t->m);
      for (auto& [num, q] : t->queues) if (num == A32(2)) { rc = CELL_EBUSY; return true; }
      t->queues.emplace_back(A32(2), A32(1));
      return true;
    }
    case 194: {  // unbind_queue(thread, spuq_num)
      auto t = thread_arg(c);
      if (!t) { rc = CELL_ESRCH; return true; }
      std::lock_guard<std::mutex> lk(t->m);
      std::erase_if(t->queues, [&](auto& p) { return p.first == A32(1); });
      return true;
    }
    case 251: {  // sys_spu_thread_group_connect_event_all_threads(group, eq, u64 req, u8* spup)
      auto g = K().get_object<SpuGroup>(A32(0));
      if (!g) { rc = CELL_ESRCH; return true; }
      const uint64_t req = ARG(2);
      for (uint32_t port = 0; port < 64; ++port) {
        if (!(req >> port & 1)) continue;
        bool free = true;
        for (auto& t : g->threads) if (t && t->ports[port]) free = false;
        if (!free) continue;
        for (auto& t : g->threads) if (t) t->ports[port] = A32(1);
        G(A32(3))[0] = uint8_t(port);
        return true;
      }
      rc = CELL_EISCONN;
      return true;
    }
    case 252: {  // disconnect_event_all_threads(group, spup)
      auto g = K().get_object<SpuGroup>(A32(0));
      if (!g) { rc = CELL_ESRCH; return true; }
      if (A32(1) < 64) for (auto& t : g->threads) if (t) t->ports[A32(1)] = 0;
      return true;
    }
    default: return false;
  }
}

// ---- sys_spu_image_import / close (sysPrxForUser) ------------------------------
namespace {

// Builds a sys_spu_image from an SPU ELF in guest memory: one COPY segment per
// PT_LOAD (plus a FILL segment for its .bss tail).
void sys_spu_image_import(ppu::Context& c) {
  const uint32_t img = A32(0), src = A32(1);
  const uint8_t* e = G(src);
  if (std::memcmp(e, "\x7F" "ELF", 4) != 0 || e[4] != 1 /* ELFCLASS32 */) { c.r[3] = CELL_ENOEXEC; return; }
  const auto be16 = [&](uint32_t o) { return uint16_t(e[o] << 8 | e[o + 1]); };
  const auto be32 = [&](uint32_t o) { return uint32_t(e[o]) << 24 | uint32_t(e[o + 1]) << 16 | uint32_t(e[o + 2]) << 8 | e[o + 3]; };
  const uint32_t entry = be32(24), phoff = be32(28);
  const uint16_t phentsize = be16(42), phnum = be16(44);
  std::vector<SpuSegment> segs;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint32_t ph = phoff + uint32_t(i) * phentsize;
    if (be32(ph) != 1) continue;  // PT_LOAD
    const uint32_t off = be32(ph + 4), vaddr = be32(ph + 8), filesz = be32(ph + 16), memsz = be32(ph + 20);
    if (filesz) segs.push_back(SpuSegment{kSegCopy, vaddr, int32_t(filesz), src + off});
    if (memsz > filesz) {
      const uint32_t start = (vaddr + filesz + 3) & ~3u;
      const uint32_t end = vaddr + memsz;
      if (end > start) segs.push_back(SpuSegment{kSegFill, start, int32_t(end - start), 0});
    }
  }
  const uint32_t table = K().hle.alloc(uint32_t(16 * std::max<size_t>(segs.size(), 1)), 16);
  for (size_t i = 0; i < segs.size(); ++i) {
    const uint32_t a = table + uint32_t(16 * i);
    K().w32(a, uint32_t(segs[i].type)); K().w32(a + 4, segs[i].ls);
    K().w32(a + 8, uint32_t(segs[i].size)); K().w32(a + 12, segs[i].addr_or_value);
  }
  K().w32(img, 1);  // SYS_SPU_IMAGE_TYPE_USER
  K().w32(img + 4, entry);
  K().w32(img + 8, table);
  K().w32(img + 12, uint32_t(segs.size()));
  c.r[3] = CELL_OK;
}
void sys_spu_image_close(ppu::Context& c) {
  if (const uint32_t t = K().r32(A32(0) + 8)) K().hle.free(t);
  c.r[3] = CELL_OK;
}

}  // namespace

void register_spu_modules() {
  hle::register_function("sysPrxForUser", "sys_spu_image_import", sys_spu_image_import);
  hle::register_function("sysPrxForUser", "sys_spu_image_close", sys_spu_image_close);
}

}  // namespace lv2
