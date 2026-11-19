// cellSpurs: the SPU runtime system. The PPU-side API is implemented here, and
// each SPURS SPU runs a small C++ kernel (instead of the firmware's SPU kernel)
// that schedules workloads and runs their policy modules in the SPU interpreter.
//
// A policy module sees what the firmware kernel provides: the kernel context at
// LS 0x100, the module loaded at LS 0xA00, and two kernel entry points it may
// branch to: 0x290 (select workload / poll) and 0x808 (exit to kernel). Those two
// addresses hold `stop` traps that land in this file.
//
// CellSpurs field offsets and the workload-selection rules follow the community's
// documentation of the SPURS data structures.

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <thread>

#include "hle.h"
#include "lv2.h"
#include "lv2_spu.h"

namespace hle {
namespace {

using lv2::Kernel;
Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }
uint8_t* G(uint32_t a) { return K().base() + a; }

// ---- error codes ----
constexpr uint32_t kErrInval = 0x80410702, kErrAgain = 0x80410701, kErrSrch = 0x80410705, kErrBusy = 0x8041070A,
                   kErrStat = 0x8041070F, kErrNull = 0x80410711;

// ---- CellSpurs layout (SPURS1, 4 KB) ----
constexpr uint32_t oReady1 = 0x00, oIdle = 0x10, oCurCont = 0x20, oPendCont = 0x30, oMinCont = 0x40, oMaxCont = 0x50,
                   oWklFlag = 0x6C, oSignal1 = 0x70, oSysSrvMsg = 0x72, oFlags1 = 0x74, oNSpus = 0x76,
                   oFlagReceiver = 0x77, oState1 = 0x80, oStatus1 = 0x90, oEvent1 = 0xA0, oEnabled = 0xB0,
                   oWklInfo1 = 0xB00, oSpuTG = 0xD30, oSpus = 0xD34, oEventQueue = 0xD5C, oFlags = 0xD80,
                   oSpuPrio = 0xD84, oPpuPrio = 0xD88, oRevision = 0xDA0, oSdk = 0xDA4, oWklH1 = 0xE00;
constexpr uint32_t kMaxWkl = 16, kSysSrvWid = 32, kMaxSpu = 8;
constexpr uint8_t kStateNone = 0, kStateRunnable = 2, kStateShuttingDown = 3, kStateRemovable = 4;
constexpr uint32_t kImgSysSrv = 0x100, kImgTasksetPm = 0x200;

// Kernel context in LS (SpursKernelContext at 0x100).
constexpr uint32_t cLocCont = 0x180, cLocPend = 0x190, cPriority = 0x1A0, cSpurs = 0x1C0, cSpuNum = 0x1C8,
                   cDmaTag = 0x1CC, cWklAddr = 0x1D0, cWklUid = 0x1D8, cWklId = 0x1DC, cExitAddr = 0x1E0,
                   cSelectAddr = 0x1E4, cModuleId = 0x1E8, cSpuIdling = 0x1EB, cRunnable1 = 0x1EC, cUniqueId = 0x220,
                   cGuid = 0x280;
constexpr uint32_t kExitAddr = 0x808, kSelectAddr = 0x290, kPmAddr = 0xA00;
constexpr uint32_t kTrapExit = 0x3801, kTrapSelect = 0x3802;

struct Spurs {
  uint32_t addr = 0;
  uint32_t nspus = 0;
  std::shared_ptr<lv2::SpuGroup> group;
};
std::mutex g_m;
std::map<uint32_t, std::shared_ptr<Spurs>> g_spurs;

std::shared_ptr<Spurs> find(uint32_t a) {
  std::lock_guard<std::mutex> lk(g_m);
  auto it = g_spurs.find(a);
  return it == g_spurs.end() ? nullptr : it->second;
}

uint32_t wbit(uint32_t wid) { return 0x80000000u >> wid; }

// ---- LS helpers ----
uint8_t* LS(lv2::SpuThread& t) { return t.core.ls(); }
void ls32(lv2::SpuThread& t, uint32_t a, uint32_t v) { t.core.set_ls32(a, v); }
uint32_t rls32(lv2::SpuThread& t, uint32_t a) { return t.core.ls32(a); }
void ls64(lv2::SpuThread& t, uint32_t a, uint64_t v) { ls32(t, a, uint32_t(v >> 32)); ls32(t, a + 4, uint32_t(v)); }
uint64_t rls64(lv2::SpuThread& t, uint32_t a) { return uint64_t(rls32(t, a)) << 32 | rls32(t, a + 4); }

// Workload selection, as the SPURS1 kernel does it. isPoll: called by a running
// module through 0x290 (the shared state is then not committed). Returns
// (wid << 32) | pollStatus; wid 32 = nothing to run.
uint64_t select_workload(lv2::SpuThread& t, bool isPoll) {
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t* ls = LS(t);
  const uint32_t sp = rls32(t, cSpurs + 4);
  uint8_t* s = G(sp);
  const uint32_t spuNum = rls32(t, cSpuNum);
  const uint32_t cur = rls32(t, cWklId);
  const uint32_t enabled = K().r32(sp + oEnabled);

  uint8_t contention[kMaxWkl], pending[kMaxWkl];
  for (uint32_t i = 0; i < kMaxWkl; ++i) {
    contention[i] = uint8_t(s[oCurCont + i] - ls[cLocCont + i]);
    pending[i] = uint8_t(s[oPendCont + i] - ls[cLocPend + i]);
    if (isPoll && i != cur) contention[i] = uint8_t(contention[i] + pending[i]);
  }

  uint32_t sel = kSysSrvWid, poll = 0;
  uint16_t best = 0;
  const uint16_t signal1 = K().r16(sp + oSignal1);
  const uint32_t wflag = K().r32(sp + oWklFlag);
  for (uint32_t i = 0; i < kMaxWkl; ++i) {
    const uint32_t info = sp + oWklInfo1 + 32 * i;
    const uint32_t img = uint32_t(K().r64(info));
    const bool runnable = (enabled & wbit(i)) && s[oState1 + i] == kStateRunnable && img != kImgTasksetPm && img != kImgSysSrv;
    const uint8_t prio = G(info + 0x18)[spuNum & 7];
    ls[cPriority + i] = prio;
    const bool sig = signal1 & (0x8000 >> i);
    const bool flag = wflag == 0 && s[oFlagReceiver] == i;
    const uint8_t ready = std::min<uint8_t>(s[oReady1 + i], kMaxSpu);
    const uint8_t idle = std::min<uint8_t>(s[oIdle + i], kMaxSpu);
    const uint8_t request = uint8_t(ready + idle);
    if (!runnable || prio == 0 || s[oMaxCont + i] <= contention[i]) continue;
    if (!(flag || sig || (ready != 0 && request > contention[i]))) continue;
    uint16_t w = (flag || sig || ready > contention[i]) ? 0x8000 : 0;
    w |= uint16_t((prio & 0x7F) << 8);
    w |= i == cur ? 0x80 : 0;
    w |= (contention[i] > 0 && s[oMinCont + i] > contention[i]) ? 0x40 : 0;
    w |= uint16_t(((kMaxSpu - contention[i]) & 0x0F) << 2);
    w |= ls[cUniqueId + i] == cur ? 0x02 : 0;
    w |= 0x01;
    if (w > best) {
      best = w;
      sel = i;
      poll = (ready > contention[i] ? 1u : 0u) | (sig ? 2u : 0u) | (flag ? 4u : 0u);
    }
  }
  ls[cSpuIdling] = sel == kSysSrvWid ? 1 : 0;

  if (sel != kSysSrvWid && (!isPoll || sel == cur)) {
    K().w16(sp + oSignal1, uint16_t(signal1 & ~(0x8000 >> sel)));
    if (sel == s[oFlagReceiver]) K().w32(sp + oWklFlag, 0xFFFFFFFF);
  }

  if (!isPoll) {
    if (sel != kSysSrvWid) ++contention[sel];
    for (uint32_t i = 0; i < kMaxWkl; ++i) {
      s[oCurCont + i] = contention[i];
      s[oPendCont + i] = uint8_t(s[oPendCont + i] - ls[cLocPend + i]);
      ls[cLocCont + i] = 0;
      ls[cLocPend + i] = 0;
    }
    if (sel != kSysSrvWid) ls[cLocCont + sel] = 1;
    ls32(t, cWklId, sel);
  } else if (sel != cur) {
    if (sel != kSysSrvWid) {
      ++pending[sel];
      for (uint32_t i = 0; i < kMaxWkl; ++i) {
        s[oPendCont + i] = uint8_t(pending[i] + ls[cLocPend + i]);
        ls[cLocPend + i] = 0;
      }
      ls[cLocPend + sel] = 1;
    }
  }
  return uint64_t(sel) << 32 | poll;
}

// Workloads whose module has exited while shutting down become removable once no
// SPU runs them any more.
void settle_shutdowns(uint32_t sp) {
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t* s = G(sp);
  for (uint32_t i = 0; i < kMaxWkl; ++i)
    if (s[oState1 + i] == kStateShuttingDown && s[oCurCont + i] == 0) s[oState1 + i] = kStateRemovable;
}

// Per-SPU kernel loop.
void kernel_main(lv2::SpuThread& t, uint32_t sp, uint32_t spuNum) {
  uint8_t* ls = LS(t);
  std::memset(ls + 0x100, 0, 0x190);
  ls64(t, cSpurs, sp);
  ls32(t, cSpuNum, spuNum);
  ls32(t, cDmaTag, 31);
  ls32(t, cWklUid, 0x20);
  ls32(t, cWklId, kSysSrvWid);
  ls32(t, cExitAddr, kExitAddr);
  ls32(t, cSelectAddr, kSelectAddr);
  ls32(t, 0x1F0, 0xF0020000);
  ls32(t, 0x200, 0x20000);
  ls32(t, cGuid, 0x423A3A02); ls32(t, cGuid + 4, 0x43F43A82); ls32(t, cGuid + 8, 0x43F26502); ls32(t, cGuid + 12, 0x420EB382);
  // Kernel entry points: traps into this file.
  ls32(t, kSelectAddr, kTrapSelect);              // stop 0x3802
  ls32(t, kSelectAddr + 4, 0x35000000);           // bi $0
  ls32(t, kExitAddr, kTrapExit);                  // stop 0x3801

  t.hle_stop = [](lv2::SpuThread& th, uint32_t code, bool& cont) {
    if (code == kTrapSelect) {
      const bool isPoll = th.core.gpr[3].w(0) != 0;
      th.core.gpr[3] = spu::V128::zero();
      th.core.gpr[3].set_d(0, select_workload(th, isPoll));
      cont = true;
      return true;
    }
    if (code == kTrapExit) { cont = false; return true; }
    return false;
  };

  uint32_t idle_spins = 0;
  for (;;) {
    t.check_stop();
    settle_shutdowns(sp);
    const uint64_t r = select_workload(t, false);
    const uint32_t wid = uint32_t(r >> 32), poll = uint32_t(r);
    if (wid >= kMaxWkl) {
      // Idle: back off gently so idle SPUs cost little host CPU.
      if (++idle_spins < 64) std::this_thread::yield();
      else std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }
    idle_spins = 0;

    // Dispatch: workload info to LS 0x3FFE0, module image to 0xA00 if not loaded.
    const uint32_t info = sp + oWklInfo1 + 32 * wid;
    std::memcpy(ls + 0x3FFE0, G(info), 0x20);
    const uint64_t img = K().r64(info), arg = K().r64(info + 8);
    const uint32_t size = K().r32(info + 0x10);
    if (rls64(t, cWklAddr) != img) {
      const uint32_t n = std::min<uint32_t>(size, 0x3FFE0 - kPmAddr);
      std::memcpy(ls + kPmAddr, G(uint32_t(img)), n);
      ls64(t, cWklAddr, img);
      ls32(t, cWklUid, G(info)[0x14]);
    }
    ls[cModuleId] = ls[cModuleId + 1] = 0;
    for (auto& reg : t.core.gpr) reg = spu::V128::zero();
    t.core.gpr[0].set_w(0, kExitAddr);
    t.core.gpr[1].set_w(0, 0x3FFB0);
    t.core.gpr[3].set_w(0, 0x100);
    t.core.gpr[4].set_d(0, arg);
    t.core.gpr[5].set_w(0, poll);
    t.core.pc = kPmAddr;
    t.core.run();  // returns when the module exits to the kernel
  }
}

// ---- attributes ---------------------------------------------------------------
// CellSpursAttribute: revision 0, sdk 4, nSpus 8, spuPrio 0xC, ppuPrio 0x10,
// exitIfNoWork 0x14, prefix 0x15[15], prefixSize 0x24, flags 0x28, container 0x2C
void _cellSpursAttributeInitialize(ppu::Context& c) {
  // (attr, revision, sdkVersion, nSpus, spuPriority, ppuPriority, exitIfNoWork)
  const uint32_t a = a32(c, 0);
  if (!a) return ret(c, kErrNull);
  std::memset(G(a), 0, 512);
  K().w32(a, a32(c, 1)); K().w32(a + 4, a32(c, 2)); K().w32(a + 8, a32(c, 3));
  K().w32(a + 0xC, a32(c, 4)); K().w32(a + 0x10, a32(c, 5)); G(a)[0x14] = c.r[9] ? 1 : 0;
  ret(c, 0);
}
void cellSpursAttributeSetNamePrefix(ppu::Context& c) {
  const uint32_t a = a32(c, 0), n = std::min<uint32_t>(a32(c, 2), 15);
  std::memcpy(G(a + 0x15), G(a32(c, 1)), n);
  K().w32(a + 0x24, n);
  ret(c, 0);
}
void ok(ppu::Context& c) { ret(c, 0); }

// ---- instance ----------------------------------------------------------------
uint32_t initialize(uint32_t sp, uint32_t nspus, int32_t spuPrio, int32_t ppuPrio, bool exitIfNoWork, std::string prefix) {
  if (!sp) return kErrNull;
  if (nspus == 0 || nspus > 6) return kErrInval;
  std::memset(G(sp), 0, 0x1000);
  uint8_t* s = G(sp);
  s[oNSpus] = uint8_t(nspus);
  s[oFlags1] = exitIfNoWork ? 0x80 : 0;
  s[oFlagReceiver] = 0xFF;
  K().w32(sp + oWklFlag, 0xFFFFFFFF);
  K().w32(sp + oSpuPrio, uint32_t(spuPrio));
  K().w32(sp + oPpuPrio, uint32_t(ppuPrio));
  K().w64(sp + 0xD00, kImgSysSrv);  // wklInfoSysSrv.addr

  auto inst = std::make_shared<Spurs>();
  inst->addr = sp;
  inst->nspus = nspus;
  inst->group = lv2::spu_group_create(prefix + "CellSpursKernelGroup", nspus, spuPrio);
  K().w32(sp + oSpuTG, inst->group->id);
  for (uint32_t i = 0; i < nspus; ++i) {
    auto t = lv2::spu_thread_create(inst->group, i, prefix + "CellSpursKernel" + std::to_string(i));
    t->hle_main = [sp, i](lv2::SpuThread& th) { kernel_main(th, sp, i); };
    K().w32(sp + oSpus + 4 * i, t->id);
  }
  {
    std::lock_guard<std::mutex> lk(g_m);
    g_spurs[sp] = inst;
  }
  lv2::spu_group_start(inst->group->id);
  K().logf("cellSpurs: initialised at 0x%08X with %u SPUs", sp, nspus);
  return 0;
}

void cellSpursInitialize(ppu::Context& c) {  // (spurs, nSpu, spuPriority, ppuPriority, exitIfNoWork)
  ret(c, initialize(a32(c, 0), a32(c, 1), int32_t(c.r[5]), int32_t(c.r[6]), c.r[7] != 0, ""));
}
void cellSpursInitializeWithAttribute(ppu::Context& c) {  // (spurs, attr)
  const uint32_t a = a32(c, 1);
  if (!a) return ret(c, kErrNull);
  const std::string prefix(reinterpret_cast<const char*>(G(a + 0x15)), std::min<uint32_t>(K().r32(a + 0x24), 15));
  ret(c, initialize(a32(c, 0), K().r32(a + 8), int32_t(K().r32(a + 0xC)), int32_t(K().r32(a + 0x10)), G(a)[0x14] != 0, prefix));
}
void cellSpursFinalize(ppu::Context& c) {
  auto inst = find(a32(c, 0));
  if (!inst) return ret(c, kErrSrch);
  lv2::spu_group_finish(*inst->group, lv2::kJoinTerminated, 0);
  lv2::spu_group_join(inst->group->id, nullptr, nullptr);
  std::lock_guard<std::mutex> lk(g_m);
  g_spurs.erase(inst->addr);
  ret(c, 0);
}
void cellSpursGetSpuThreadId(ppu::Context& c) {  // (spurs, u32* thread, u32* nThreads)
  auto inst = find(a32(c, 0));
  if (!inst) return ret(c, kErrSrch);
  const uint32_t n = std::min(K().r32(a32(c, 2)), inst->nspus);
  for (uint32_t i = 0; i < n; ++i) K().w32(a32(c, 1) + 4 * i, inst->group->threads[i]->id);
  K().w32(a32(c, 2), n);
  ret(c, 0);
}
void cellSpursGetSpuThreadGroupId(ppu::Context& c) {
  auto inst = find(a32(c, 0));
  if (!inst) return ret(c, kErrSrch);
  K().w32(a32(c, 1), inst->group->id);
  ret(c, 0);
}
void cellSpursGetNumSpuThread(ppu::Context& c) {
  auto inst = find(a32(c, 0));
  if (!inst) return ret(c, kErrSrch);
  K().w32(a32(c, 1), inst->nspus);
  ret(c, 0);
}
// (spurs, queue, u8* port, s32 isDynamic): connects the queue to an SPU port on every kernel thread.
void cellSpursAttachLv2EventQueue(ppu::Context& c) {
  auto inst = find(a32(c, 0));
  if (!inst) return ret(c, kErrSrch);
  const uint32_t q = a32(c, 1), pp = a32(c, 2);
  uint32_t port = c.r[6] ? 0x10 : G(pp)[0];
  if (c.r[6]) {
    for (; port < 63; ++port) {
      bool free = true;
      for (auto& t : inst->group->threads) if (t && t->ports[port]) free = false;
      if (free) break;
    }
  }
  if (port > 63) return ret(c, kErrInval);
  for (auto& t : inst->group->threads) if (t) t->ports[port] = q;
  G(pp)[0] = uint8_t(port);
  ret(c, 0);
}
void cellSpursDetachLv2EventQueue(ppu::Context& c) {  // (spurs, port)
  auto inst = find(a32(c, 0));
  if (!inst) return ret(c, kErrSrch);
  const uint32_t port = a32(c, 1) & 63;
  for (auto& t : inst->group->threads) if (t) t->ports[port] = 0;
  ret(c, 0);
}

// ---- workloads ---------------------------------------------------------------
// CellSpursWorkloadAttribute: revision 0, sdk 4, pm 8, size 0xC, data 0x10,
// priority[8] 0x18, minContention 0x20, maxContention 0x24, nameClass 0x28,
// nameInstance 0x2C, hook 0x30, hookArg 0x34
void _cellSpursWorkloadAttributeInitialize(ppu::Context& c) {
  // (attr, revision, sdk, pm, size, u64 data, const u8 priority[8], minCnt, maxCnt [stack])
  const uint32_t a = a32(c, 0);
  if (!a) return ret(c, kErrNull);
  std::memset(G(a), 0, 512);
  K().w32(a, a32(c, 1)); K().w32(a + 4, a32(c, 2)); K().w32(a + 8, a32(c, 3)); K().w32(a + 0xC, a32(c, 4));
  K().w64(a + 0x10, c.r[8]);
  if (a32(c, 6)) std::memcpy(G(a + 0x18), G(a32(c, 6)), 8);
  K().w32(a + 0x20, a32(c, 7));
  K().w32(a + 0x24, uint32_t(K().r64(uint32_t(c.r[1]) + 112)));  // 9th argument: parameter save area
  ret(c, 0);
}
void cellSpursWorkloadAttributeSetName(ppu::Context& c) {
  K().w32(a32(c, 0) + 0x28, a32(c, 1));
  K().w32(a32(c, 0) + 0x2C, a32(c, 2));
  ret(c, 0);
}

uint32_t add_workload(uint32_t sp, uint32_t widp, uint32_t pm, uint32_t size, uint64_t data, const uint8_t prio[8],
                      uint32_t minCnt, uint32_t maxCnt, uint32_t nameClass, uint32_t nameInstance) {
  auto inst = find(sp);
  if (!inst) return kErrSrch;
  if (!widp) return kErrNull;
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t* s = G(sp);
  const uint32_t enabled = K().r32(sp + oEnabled);
  uint32_t wid = 0;
  while (wid < kMaxWkl && (enabled & wbit(wid))) ++wid;
  if (wid == kMaxWkl) return kErrAgain;
  const uint32_t info = sp + oWklInfo1 + 32 * wid;
  // Workloads sharing an image share a unique id (the module stays loaded).
  uint8_t uid = uint8_t(wid);
  for (uint32_t i = 0; i < kMaxWkl; ++i)
    if ((enabled & wbit(i)) && uint32_t(K().r64(sp + oWklInfo1 + 32 * i)) == pm) uid = G(sp + oWklInfo1 + 32 * i)[0x14];
  K().w64(info, pm);
  K().w64(info + 8, data);
  K().w32(info + 0x10, size);
  G(info)[0x14] = uid;
  std::memcpy(G(info + 0x18), prio, 8);
  s[oMinCont + wid] = uint8_t(std::min(minCnt, inst->nspus));
  s[oMaxCont + wid] = uint8_t(std::min<uint32_t>(maxCnt ? maxCnt : kMaxSpu, kMaxSpu));
  s[oReady1 + wid] = 0;
  s[oIdle + wid] = 0;
  s[oCurCont + wid] = 0;
  s[oPendCont + wid] = 0;
  s[oStatus1 + wid] = 0;
  s[oEvent1 + wid] = 0;
  K().w64(sp + oWklH1 + 16 * wid, nameClass);
  K().w64(sp + oWklH1 + 16 * wid + 8, nameInstance);
  s[oState1 + wid] = kStateRunnable;
  K().w32(sp + oEnabled, enabled | wbit(wid));
  K().w32(widp, wid);
  K().logf("cellSpurs: workload %u added (module 0x%08X, %u bytes, contention %u..%u)", wid, pm, size, minCnt, maxCnt);
  return 0;
}

void cellSpursAddWorkloadWithAttribute(ppu::Context& c) {  // (spurs, u32* wid, attr)
  const uint32_t a = a32(c, 2);
  if (!a) return ret(c, kErrNull);
  ret(c, add_workload(a32(c, 0), a32(c, 1), K().r32(a + 8), K().r32(a + 0xC), K().r64(a + 0x10), G(a + 0x18),
                      K().r32(a + 0x20), K().r32(a + 0x24), K().r32(a + 0x28), K().r32(a + 0x2C)));
}
void cellSpursAddWorkload(ppu::Context& c) {  // (spurs, wid*, pm, size, u64 data, priority*, min, max)
  uint8_t prio[8] = {};
  if (a32(c, 5)) std::memcpy(prio, G(a32(c, 5)), 8);
  ret(c, add_workload(a32(c, 0), a32(c, 1), a32(c, 2), a32(c, 3), c.r[7], prio, a32(c, 6), a32(c, 7), 0, 0));
}

bool valid_wid(uint32_t sp, uint32_t wid) { return wid < kMaxWkl && (K().r32(sp + oEnabled) & wbit(wid)); }

void cellSpursReadyCountStore(ppu::Context& c) {  // (spurs, wid, u32 value)
  const uint32_t sp = a32(c, 0), wid = a32(c, 1);
  if (!find(sp)) return ret(c, kErrSrch);
  if (!valid_wid(sp, wid)) return ret(c, kErrInval);
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  G(sp)[oReady1 + wid] = uint8_t(std::min<uint32_t>(a32(c, 2), 255));
  ret(c, 0);
}
void cellSpursReadyCountAdd(ppu::Context& c) {  // (spurs, wid, u32* old, s32 value)
  const uint32_t sp = a32(c, 0), wid = a32(c, 1);
  if (!valid_wid(sp, wid)) return ret(c, kErrInval);
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t& r = G(sp)[oReady1 + wid];
  K().w32(a32(c, 2), r);
  r = uint8_t(std::clamp<int32_t>(int32_t(r) + int32_t(c.r[6]), 0, 255));
  ret(c, 0);
}
void cellSpursSendWorkloadSignal(ppu::Context& c) {  // (spurs, wid)
  const uint32_t sp = a32(c, 0), wid = a32(c, 1);
  if (!valid_wid(sp, wid)) return ret(c, kErrInval);
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  K().w16(sp + oSignal1, uint16_t(K().r16(sp + oSignal1) | (0x8000 >> wid)));
  ret(c, 0);
}
void cellSpursShutdownWorkload(ppu::Context& c) {  // (spurs, wid)
  const uint32_t sp = a32(c, 0), wid = a32(c, 1);
  if (!find(sp)) return ret(c, kErrSrch);
  if (!valid_wid(sp, wid)) return ret(c, kErrInval);
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t& st = G(sp)[oState1 + wid];
  if (st != kStateRunnable) return ret(c, kErrStat);
  st = G(sp)[oCurCont + wid] == 0 ? kStateRemovable : kStateShuttingDown;
  ret(c, 0);
}
void cellSpursWaitForWorkloadShutdown(ppu::Context& c) {  // (spurs, wid)
  const uint32_t sp = a32(c, 0), wid = a32(c, 1);
  if (!find(sp)) return ret(c, kErrSrch);
  if (!valid_wid(sp, wid)) return ret(c, kErrInval);
  for (;;) {
    settle_shutdowns(sp);
    const uint8_t st = G(sp)[oState1 + wid];
    if (st == kStateRemovable) break;
    if (st != kStateShuttingDown) return ret(c, kErrStat);
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  ret(c, 0);
}
void cellSpursRemoveWorkload(ppu::Context& c) {  // (spurs, wid)
  const uint32_t sp = a32(c, 0), wid = a32(c, 1);
  if (!find(sp)) return ret(c, kErrSrch);
  if (!valid_wid(sp, wid)) return ret(c, kErrInval);
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t* s = G(sp);
  if (s[oState1 + wid] != kStateRemovable) return ret(c, kErrBusy);
  s[oState1 + wid] = kStateNone;
  K().w32(sp + oEnabled, K().r32(sp + oEnabled) & ~wbit(wid));
  ret(c, 0);
}

// ---- tasksets ----------------------------------------------------------------
// CellSpursTasksetAttribute: revision 0, sdk 4, args 8, priority[8] 0x10,
// max_contention 0x18, name 0x1C, taskset_size 0x20, enable_clear_ls 0x24
void _cellSpursTasksetAttributeInitialize(ppu::Context& c) {
  // (attr, revision, sdk, u64 args, const u8 priority[8], maxContention)
  const uint32_t a = a32(c, 0);
  if (!a) return ret(c, kErrNull);
  std::memset(G(a), 0, 512);
  K().w32(a, a32(c, 1)); K().w32(a + 4, a32(c, 2)); K().w64(a + 8, c.r[6]);
  if (a32(c, 4)) std::memcpy(G(a + 0x10), G(a32(c, 4)), 8);
  K().w32(a + 0x18, a32(c, 5));
  K().w32(a + 0x20, 6400);
  ret(c, 0);
}
void cellSpursTasksetAttributeSetName(ppu::Context& c) { K().w32(a32(c, 0) + 0x1C, a32(c, 1)); ret(c, 0); }
void cellSpursTasksetAttributeSetTasksetSize(ppu::Context& c) { K().w32(a32(c, 0) + 0x20, a32(c, 1)); ret(c, 0); }

// A taskset is a workload whose policy module is the firmware's taskset PM. Tasks
// (SPU ELFs) are not supported yet; a taskset without tasks never needs an SPU.
uint32_t create_taskset(uint32_t sp, uint32_t ts, uint64_t args, const uint8_t prio[8], uint32_t maxCnt, uint32_t size, uint32_t name) {
  if (!ts) return kErrNull;
  std::memset(G(ts), 0, std::max<uint32_t>(size, 6400));
  K().w64(ts + 0x60, sp);
  K().w64(ts + 0x68, args);
  K().w32(ts + 0x1890, size);
  const uint32_t scratch = K().hle.alloc(4, 4);
  const uint32_t rc = add_workload(sp, scratch, kImgTasksetPm, 0, ts, prio, 1, maxCnt, 0, name);
  if (rc == 0) K().w32(ts + 0x74, K().r32(scratch));
  K().hle.free(scratch);
  return rc;
}
void cellSpursCreateTasksetWithAttribute(ppu::Context& c) {  // (spurs, taskset, attr)
  const uint32_t a = a32(c, 2);
  if (!a) return ret(c, kErrNull);
  ret(c, create_taskset(a32(c, 0), a32(c, 1), K().r64(a + 8), G(a + 0x10), K().r32(a + 0x18), K().r32(a + 0x20), K().r32(a + 0x1C)));
}
void cellSpursCreateTaskset(ppu::Context& c) {  // (spurs, taskset, u64 args, priority*, maxContention)
  uint8_t prio[8] = {};
  if (a32(c, 3)) std::memcpy(prio, G(a32(c, 3)), 8);
  ret(c, create_taskset(a32(c, 0), a32(c, 1), c.r[5], prio, a32(c, 4), 6400, 0));
}
void taskset_finish(ppu::Context& c, bool remove) {
  const uint32_t ts = a32(c, 0);
  if (!ts) return ret(c, kErrNull);
  const uint32_t sp = uint32_t(K().r64(ts + 0x60)), wid = K().r32(ts + 0x74);
  if (!find(sp) || !valid_wid(sp, wid)) return ret(c, kErrSrch);
  std::lock_guard<std::mutex> lk(lv2::spu_line_mutex());
  uint8_t* s = G(sp);
  if (s[oState1 + wid] == kStateRunnable) s[oState1 + wid] = kStateRemovable;
  if (remove && s[oState1 + wid] == kStateRemovable) {
    s[oState1 + wid] = kStateNone;
    K().w32(sp + oEnabled, K().r32(sp + oEnabled) & ~wbit(wid));
  }
  ret(c, 0);
}
void cellSpursShutdownTaskset(ppu::Context& c) { taskset_finish(c, false); }
void cellSpursJoinTaskset(ppu::Context& c) { taskset_finish(c, true); }
void cellSpursCreateTask(ppu::Context& c) {
  K().logf("cellSpursCreateTask: SPURS tasks are not supported yet");
  ret(c, kErrAgain);
}

}  // namespace

void register_spurs_module() {
#define R(fn) register_function("cellSpurs", #fn, fn)
  R(_cellSpursAttributeInitialize);
  R(cellSpursAttributeSetNamePrefix);
  register_function("cellSpurs", "cellSpursAttributeSetSpuThreadGroupType", ok);
  register_function("cellSpurs", "cellSpursAttributeEnableSpuPrintfIfAvailable", ok);
  register_function("cellSpurs", "cellSpursAttributeSetMemoryContainerForSpuThread", ok);
  register_function("cellSpurs", "cellSpursTaskExitCodeInitialize", ok);
  register_function("cellSpurs", "cellSpursSetMaxContention", ok);
  register_function("cellSpurs", "cellSpursSetPriorities", ok);
  R(cellSpursInitialize);
  R(cellSpursInitializeWithAttribute);
  R(cellSpursFinalize);
  R(cellSpursGetSpuThreadId);
  R(cellSpursGetSpuThreadGroupId);
  R(cellSpursGetNumSpuThread);
  R(cellSpursAttachLv2EventQueue);
  R(cellSpursDetachLv2EventQueue);
  R(_cellSpursWorkloadAttributeInitialize);
  R(cellSpursWorkloadAttributeSetName);
  R(cellSpursAddWorkloadWithAttribute);
  R(cellSpursAddWorkload);
  R(cellSpursReadyCountStore);
  R(cellSpursReadyCountAdd);
  R(cellSpursSendWorkloadSignal);
  R(cellSpursShutdownWorkload);
  R(cellSpursWaitForWorkloadShutdown);
  R(cellSpursRemoveWorkload);
  R(_cellSpursTasksetAttributeInitialize);
  R(cellSpursTasksetAttributeSetName);
  R(cellSpursTasksetAttributeSetTasksetSize);
  R(cellSpursCreateTasksetWithAttribute);
  R(cellSpursCreateTaskset);
  R(cellSpursShutdownTaskset);
  R(cellSpursJoinTaskset);
  R(cellSpursCreateTask);
#undef R
}

}  // namespace hle
