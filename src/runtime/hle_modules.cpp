// Built-in HLE modules: the pieces of the PS3 firmware libraries a game needs to
// get through startup. Graphics (cellGcm), audio, input, file system and network
// are separate stages; their imports currently resolve to logging stubs.

// Guest strcpy/strcat are unbounded by definition; they operate on guest memory.
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <vector>

#include "hle.h"
#include "lv2.h"

namespace hle {
namespace {

using lv2::Kernel;
using namespace lv2;

Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }

// ============================================================================
// sysPrxForUser
// ============================================================================
void sys_initialize_tls(ppu::Context& c) {
  // (thread_id, tls_seg_addr, tls_seg_size, tls_mem_size)
  K().set_tls_image(a32(c, 1), a32(c, 2), a32(c, 3));
  c.r[13] = K().alloc_tls();
  K().current().tls_block = uint32_t(c.r[13]);
  ret(c, CELL_OK);
}

void sys_process_exit(ppu::Context& c) { throw ProcessExit{int(int32_t(c.r[3]))}; }
void sys_process_atexitspawn(ppu::Context& c) { ret(c, CELL_OK); }
void sys_process_is_stack(ppu::Context& c) {
  const uint32_t a = a32(c, 0);
  ret(c, a >= kStackBase && a < kStackBase + kStackSize ? 1 : 0);
}
void sys_prx_exitspawn_with_level(ppu::Context& c) { ret(c, CELL_OK); }

void sys_ppu_thread_create(ppu::Context& c) {
  // (u64* thread_id, u32 entry_opd, u64 arg, s32 prio, u32 stacksize, u64 flags, const char* name)
  auto t = K().create_thread(a32(c, 1), c.r[5], int32_t(c.r[6]), a32(c, 4), (c.r[8] & 1) != 0,
                             K().read_str(a32(c, 6), 27));
  if (!t) return ret(c, CELL_ENOMEM);
  K().w64(a32(c, 0), t->id);
  ret(c, K().start_thread(t->id));
}
void sys_ppu_thread_exit(ppu::Context& c) { throw ThreadExit{c.r[3]}; }
void sys_ppu_thread_get_id(ppu::Context& c) {
  K().w64(a32(c, 0), K().current().id);
  ret(c, CELL_OK);
}

void sys_time_get_system_time(ppu::Context& c) {
  ret(c, lv2::system_time_us());
}

// sys_lwmutex_t: +0 u32 owner, +4 u32 waiter, +8 u32 attribute, +12 u32 recursive_count,
//                +16 u32 sleep_queue (we keep the kernel mutex id here), +20 pad
constexpr uint32_t kLwFree = 0xFFFFFFFF;
void sys_lwmutex_create(ppu::Context& c) {
  const uint32_t lw = a32(c, 0), attr = a32(c, 1);
  const uint32_t protocol = attr ? K().r32(attr) : 2, recursive = attr ? K().r32(attr + 4) : 0x20;
  uint32_t id = 0;
  mutex_create(recursive == 0x10, &id);
  K().w32(lw, kLwFree);
  K().w32(lw + 4, 0);
  K().w32(lw + 8, protocol | recursive);
  K().w32(lw + 12, 0);
  K().w32(lw + 16, id);
  ret(c, CELL_OK);
}
void lw_sync_owner(uint32_t lw) {
  auto m = K().get_object<Mutex>(K().r32(lw + 16));
  if (!m) return;
  std::lock_guard<std::mutex> lk(m->m);
  K().w32(lw, m->owner ? uint32_t(m->owner) : kLwFree);
  K().w32(lw + 12, m->count);
}
void sys_lwmutex_lock(ppu::Context& c) {
  const uint32_t lw = a32(c, 0);
  const uint32_t rc = mutex_lock(K().r32(lw + 16), c.r[4]);
  if (rc == CELL_OK) lw_sync_owner(lw);
  ret(c, rc);
}
void sys_lwmutex_trylock(ppu::Context& c) {
  const uint32_t lw = a32(c, 0);
  const uint32_t rc = mutex_trylock(K().r32(lw + 16));
  if (rc == CELL_OK) lw_sync_owner(lw);
  ret(c, rc);
}
void sys_lwmutex_unlock(ppu::Context& c) {
  const uint32_t lw = a32(c, 0);
  const uint32_t rc = mutex_unlock(K().r32(lw + 16));
  if (rc == CELL_OK) lw_sync_owner(lw);
  ret(c, rc);
}
void sys_lwmutex_destroy(ppu::Context& c) {
  const uint32_t lw = a32(c, 0);
  const uint32_t rc = mutex_destroy(K().r32(lw + 16));
  if (rc == CELL_OK) K().w32(lw + 16, 0);
  ret(c, rc);
}

// sys_lwcond_t: +0 u32 lwmutex, +4 u32 lwcond_queue (kernel cond id)
void sys_lwcond_create(ppu::Context& c) {
  const uint32_t lc = a32(c, 0), lw = a32(c, 1);
  uint32_t id = 0;
  const uint32_t rc = cond_create(K().r32(lw + 16), &id);
  if (rc == CELL_OK) { K().w32(lc, lw); K().w32(lc + 4, id); }
  ret(c, rc);
}
void sys_lwcond_destroy(ppu::Context& c) { ret(c, cond_destroy(K().r32(a32(c, 0) + 4))); }
void sys_lwcond_signal(ppu::Context& c) { ret(c, cond_signal(K().r32(a32(c, 0) + 4))); }
void sys_lwcond_signal_all(ppu::Context& c) { ret(c, cond_signal_all(K().r32(a32(c, 0) + 4))); }
void sys_lwcond_signal_to(ppu::Context& c) { ret(c, cond_signal_to(K().r32(a32(c, 0) + 4), c.r[4])); }
void sys_lwcond_wait(ppu::Context& c) {
  const uint32_t lc = a32(c, 0);
  const uint32_t rc = cond_wait(K().r32(lc + 4), c.r[4]);
  lw_sync_owner(K().r32(lc));
  ret(c, rc);
}

// ---- libc-style helpers ----
uint8_t* G(uint32_t a) { return K().base() + a; }
void _sys_malloc(ppu::Context& c) { ret(c, K().heap.alloc(a32(c, 0), 16)); }
void _sys_memalign(ppu::Context& c) { ret(c, K().heap.alloc(a32(c, 1), a32(c, 0))); }
void _sys_free(ppu::Context& c) { if (a32(c, 0)) K().heap.free(a32(c, 0)); ret(c, 0); }
void _sys_memset(ppu::Context& c) { std::memset(G(a32(c, 0)), int(c.r[4]), a32(c, 2)); }
void _sys_memcpy(ppu::Context& c) { std::memcpy(G(a32(c, 0)), G(a32(c, 1)), a32(c, 2)); }
void _sys_memmove(ppu::Context& c) { std::memmove(G(a32(c, 0)), G(a32(c, 1)), a32(c, 2)); }
void _sys_memcmp(ppu::Context& c) { ret(c, uint64_t(int64_t(std::memcmp(G(a32(c, 0)), G(a32(c, 1)), a32(c, 2))))); }
void _sys_memchr(ppu::Context& c) {
  const void* p = std::memchr(G(a32(c, 0)), int(c.r[4]), a32(c, 2));
  ret(c, p ? uint64_t(static_cast<const uint8_t*>(p) - K().base()) : 0);
}
// _sys_heap_*: small guest heaps; backed by the kernel heap (ids are just counters).
void _sys_heap_create_heap(ppu::Context& c) {
  static std::atomic<uint32_t> next{1};
  ret(c, next++);
}
void _sys_heap_delete_heap(ppu::Context& c) { ret(c, CELL_OK); }
void _sys_heap_malloc(ppu::Context& c) { ret(c, K().heap.alloc(a32(c, 1), 16)); }          // (heap, size)
void _sys_heap_memalign(ppu::Context& c) { ret(c, K().heap.alloc(a32(c, 2), a32(c, 1))); }  // (heap, align, size)
void _sys_heap_free(ppu::Context& c) { if (a32(c, 1)) K().heap.free(a32(c, 1)); ret(c, CELL_OK); }  // (heap, ptr)

void _sys_strlen(ppu::Context& c) { ret(c, std::strlen(reinterpret_cast<const char*>(G(a32(c, 0))))); }
void _sys_strcmp(ppu::Context& c) {
  ret(c, uint64_t(int64_t(std::strcmp(reinterpret_cast<const char*>(G(a32(c, 0))), reinterpret_cast<const char*>(G(a32(c, 1)))))));
}
void _sys_strncmp(ppu::Context& c) {
  ret(c, uint64_t(int64_t(std::strncmp(reinterpret_cast<const char*>(G(a32(c, 0))),
                                       reinterpret_cast<const char*>(G(a32(c, 1))), a32(c, 2)))));
}
void _sys_strcpy(ppu::Context& c) {
  std::strcpy(reinterpret_cast<char*>(G(a32(c, 0))), reinterpret_cast<const char*>(G(a32(c, 1))));
}
void _sys_strncpy(ppu::Context& c) {
  std::strncpy(reinterpret_cast<char*>(G(a32(c, 0))), reinterpret_cast<const char*>(G(a32(c, 1))), a32(c, 2));
}
void _sys_strcat(ppu::Context& c) {
  std::strcat(reinterpret_cast<char*>(G(a32(c, 0))), reinterpret_cast<const char*>(G(a32(c, 1))));
}
void _sys_strchr(ppu::Context& c) {
  const char* p = std::strchr(reinterpret_cast<const char*>(G(a32(c, 0))), int(c.r[4]));
  ret(c, p ? uint64_t(reinterpret_cast<const uint8_t*>(p) - K().base()) : 0);
}
void _sys_strrchr(ppu::Context& c) {
  const char* p = std::strrchr(reinterpret_cast<const char*>(G(a32(c, 0))), int(c.r[4]));
  ret(c, p ? uint64_t(reinterpret_cast<const uint8_t*>(p) - K().base()) : 0);
}
void _sys_tolower(ppu::Context& c) { ret(c, uint64_t(std::tolower(int(c.r[3] & 0xFF)))); }
void _sys_toupper(ppu::Context& c) { ret(c, uint64_t(std::toupper(int(c.r[3] & 0xFF)))); }

void emit_tty(const std::string& s) {
  if (K().tty) K().tty(s);
  else std::fwrite(s.data(), 1, s.size(), stdout);
}
void _sys_printf(ppu::Context& c) {
  const std::string s = format_guest(c, a32(c, 0), 4);
  emit_tty(s);
  ret(c, s.size());
}
void _sys_sprintf(ppu::Context& c) {
  const std::string s = format_guest(c, a32(c, 1), 5);
  std::memcpy(G(a32(c, 0)), s.c_str(), s.size() + 1);
  ret(c, s.size());
}
void _sys_snprintf(ppu::Context& c) {
  const std::string s = format_guest(c, a32(c, 2), 6);
  const uint32_t n = a32(c, 1);
  if (n) {
    const size_t len = std::min<size_t>(s.size(), n - 1);
    std::memcpy(G(a32(c, 0)), s.data(), len);
    G(a32(c, 0))[len] = 0;
  }
  ret(c, s.size());
}

// ============================================================================
// cellSysmodule
// ============================================================================
void ok(ppu::Context& c) { ret(c, CELL_OK); }
void cellSysmoduleLoadModule(ppu::Context& c) {
  K().logf("cellSysmoduleLoadModule(0x%04X)", unsigned(c.r[3] & 0xFFFF));
  ret(c, CELL_OK);
}

// ============================================================================
// cellSysutil
// ============================================================================
struct SysutilCallback { uint32_t func = 0, userdata = 0; };
SysutilCallback g_callbacks[4];

void cellSysutilRegisterCallback(ppu::Context& c) {
  const uint32_t slot = a32(c, 0);
  if (slot >= 4) return ret(c, 0x8002B101);  // CELL_SYSUTIL_ERROR_VALUE
  g_callbacks[slot] = {a32(c, 1), a32(c, 2)};
  ret(c, CELL_OK);
}
void cellSysutilUnregisterCallback(ppu::Context& c) {
  const uint32_t slot = a32(c, 0);
  if (slot >= 4) return ret(c, 0x8002B101);
  g_callbacks[slot] = {};
  ret(c, CELL_OK);
}
// Deferred callbacks (dialog results, ...) run on the game thread that calls
// cellSysutilCheckCallback, as on the console. No XMB events are generated yet.
struct PendingCb { uint32_t opd; uint64_t a0, a1, a2; };
std::mutex g_cb_m;
std::vector<PendingCb> g_pending;
void cellSysutilCheckCallback(ppu::Context& c) {
  std::vector<PendingCb> run;
  {
    std::lock_guard<std::mutex> lk(g_cb_m);
    run.swap(g_pending);
  }
  for (auto& p : run) K().call_guest(p.opd, {p.a0, p.a1, p.a2});
  ret(c, CELL_OK);
}

// ---- cellMsgDialog: no UI; button dialogs are answered OK / YES at once ----------
// type bits 4-6: 0 = no buttons, 1 = YES/NO, 2 = OK. Callback(s32 button, void* userdata).
void cellMsgDialogOpen2(ppu::Context& c) {  // (type, msg, callback, userdata, extParam)
  const uint32_t type = a32(c, 0), cb = a32(c, 2);
  K().logf("message dialog: \"%s\"", K().read_str(a32(c, 1), 512).c_str());
  const uint32_t buttons = (type >> 4) & 7;
  if (cb && (buttons == 1 || buttons == 2)) {
    std::lock_guard<std::mutex> lk(g_cb_m);
    g_pending.push_back({cb, 1 /* YES / OK */, a32(c, 3), 0});
  }
  ret(c, CELL_OK);
}
void cellMsgDialogOpenErrorCode(ppu::Context& c) {  // (errorCode, callback, userdata, extParam)
  K().logf("error dialog: 0x%08X", a32(c, 0));
  if (const uint32_t cb = a32(c, 1)) {
    std::lock_guard<std::mutex> lk(g_cb_m);
    g_pending.push_back({cb, 1, a32(c, 2), 0});
  }
  ret(c, CELL_OK);
}

}  // namespace

// A system event for every registered sysutil callback: callback(status, param, userdata).
void queue_sysutil_event(uint64_t status, uint64_t param) {
  std::lock_guard<std::mutex> lk(g_cb_m);
  for (const auto& cb : g_callbacks)
    if (cb.func) g_pending.push_back({cb.func, status, param, cb.userdata});
}

namespace {

void cellSysutilGetSystemParamInt(ppu::Context& c) {
  uint32_t v = 0;
  switch (a32(c, 0)) {
    case 0x0111: v = 1; break;  // language: English (US)
    case 0x0112: v = 1; break;  // enter button: cross
    case 0x0114: v = 0; break;  // date format
    case 0x0115: v = 1; break;  // time format: 24h
    default: v = 0; break;
  }
  K().w32(a32(c, 1), v);
  ret(c, CELL_OK);
}
void cellSysutilGetSystemParamString(ppu::Context& c) {
  const uint32_t buf = a32(c, 1), size = a32(c, 2);
  const char* s = "Player";
  if (size) {
    const size_t n = std::min<size_t>(std::strlen(s), size - 1);
    std::memcpy(G(buf), s, n);
    G(buf)[n] = 0;
  }
  ret(c, CELL_OK);
}

// sys_mmapper wrappers: (size, u64 flags, u32* mem_id) / (addr, mem_id, flags) /
// (mem_id) / (addr, u32* mem_id)
void sys_mmapper_allocate_memory(ppu::Context& c) {
  uint32_t id = 0;
  const uint32_t rc = mmapper_allocate_memory(a32(c, 0), &id);
  if (rc == CELL_OK) K().w32(a32(c, 2), id);
  ret(c, rc);
}
void sys_mmapper_map_memory(ppu::Context& c) { ret(c, mmapper_map_memory(a32(c, 0), a32(c, 1))); }
void sys_mmapper_free_memory(ppu::Context& c) { ret(c, mmapper_free_memory(a32(c, 0))); }
void sys_mmapper_unmap_memory(ppu::Context& c) {
  uint32_t id = 0;
  const uint32_t rc = mmapper_unmap_memory(a32(c, 0), &id);
  if (rc == CELL_OK) K().w32(a32(c, 1), id);
  ret(c, rc);
}

// sys_prx_*: game-shipped PRX modules (Skate 3: EAWebkit, the in-game browser) are
// encrypted SPRX files, which are never loaded. Loading "succeeds" with a dummy
// module so startup proceeds; its imports resolve to logging stubs.
void sys_prx_load_module(ppu::Context& c) {
  static std::atomic<uint32_t> next{0x23000001};
  K().logf("sys_prx_load_module(\"%s\"): not loaded (game PRX), dummy module", K().read_str(a32(c, 0)).c_str());
  ret(c, next++);
}
void sys_prx_start_module(ppu::Context& c) {  // (id, args, argp, s32* modres, flags, opt)
  K().w32(a32(c, 3), 0);
  ret(c, CELL_OK);
}

// sys_ppu_thread_once(s32* once_ctrl, void (*init)()): runs init exactly once.
void sys_ppu_thread_once(ppu::Context& c) {
  static std::mutex m;
  const uint32_t ctrl = a32(c, 0), fn = a32(c, 1);
  std::lock_guard<std::mutex> lk(m);
  if (K().r32(ctrl) == 0) {  // SYS_PPU_THREAD_ONCE_INIT
    K().w32(ctrl, 1);
    K().call_guest(fn, {});
  }
  ret(c, CELL_OK);
}

// ============================================================================
// cellRtc. CellRtcTick = u64 microseconds since 0001-01-01 00:00 UTC.
// CellRtcDateTime = {u16 year, month, day, hour, minute, second; u32 microsecond}.
// ============================================================================
constexpr uint64_t kUnixEpochTicks = 62135596800ull * 1000000ull;

// Days since 1970-01-01 for a civil date, and back (proleptic Gregorian).
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + int64_t(doe) - 719468;
}
void civil_from_days(int64_t z, int64_t& y, unsigned& m, unsigned& d) {
  z += 719468;
  const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = unsigned(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp < 10 ? mp + 3 : mp - 9;
  y = int64_t(yoe) + era * 400 + (m <= 2);
}

uint64_t rtc_now_utc() {
  using namespace std::chrono;
  return kUnixEpochTicks +
         uint64_t(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
}
int64_t local_offset_us() {
  const std::time_t now = std::time(nullptr);
  std::tm lt{}, gt{};
  localtime_s(&lt, &now);
  gmtime_s(&gt, &now);
  gt.tm_isdst = lt.tm_isdst;
  return int64_t(std::difftime(std::mktime(&lt), std::mktime(&gt))) * 1000000;
}
void write_datetime(uint32_t p, uint64_t tick) {
  const int64_t us = int64_t(tick - kUnixEpochTicks);
  int64_t days = us / 86400000000ll, rem = us % 86400000000ll;
  if (rem < 0) { rem += 86400000000ll; --days; }
  int64_t y; unsigned m, d;
  civil_from_days(days, y, m, d);
  const uint64_t secs = uint64_t(rem) / 1000000;
  K().w16(p + 0, uint16_t(y)); K().w16(p + 2, uint16_t(m)); K().w16(p + 4, uint16_t(d));
  K().w16(p + 6, uint16_t(secs / 3600)); K().w16(p + 8, uint16_t(secs / 60 % 60));
  K().w16(p + 10, uint16_t(secs % 60)); K().w32(p + 12, uint32_t(uint64_t(rem) % 1000000));
}
uint64_t read_datetime(uint32_t p) {
  const int64_t days = days_from_civil(K().r16(p), K().r16(p + 2), K().r16(p + 4));
  const int64_t secs = days * 86400 + K().r16(p + 6) * 3600 + K().r16(p + 8) * 60 + K().r16(p + 10);
  return kUnixEpochTicks + uint64_t(secs * 1000000 + K().r32(p + 12));
}
void cellRtcGetCurrentTick(ppu::Context& c) { K().w64(a32(c, 0), rtc_now_utc()); ret(c, CELL_OK); }
void cellRtcGetCurrentClock(ppu::Context& c) {  // (DateTime*, s32 timezone minutes)
  write_datetime(a32(c, 0), rtc_now_utc() + uint64_t(int64_t(int32_t(c.r[4])) * 60000000));
  ret(c, CELL_OK);
}
void cellRtcGetCurrentClockLocalTime(ppu::Context& c) {
  write_datetime(a32(c, 0), rtc_now_utc() + uint64_t(local_offset_us()));
  ret(c, CELL_OK);
}
void cellRtcGetTick(ppu::Context& c) { K().w64(a32(c, 1), read_datetime(a32(c, 0))); ret(c, CELL_OK); }
void cellRtcSetTick(ppu::Context& c) { write_datetime(a32(c, 0), K().r64(a32(c, 1))); ret(c, CELL_OK); }
void cellRtcGetDayOfWeek(ppu::Context& c) {  // 0 = Sunday
  const int64_t days = days_from_civil(int32_t(c.r[3]), unsigned(c.r[4]), unsigned(c.r[5]));
  ret(c, uint64_t((days % 7 + 11) % 7));
}
void cellRtcConvertUtcToLocalTime(ppu::Context& c) {
  K().w64(a32(c, 1), K().r64(a32(c, 0)) + uint64_t(local_offset_us()));
  ret(c, CELL_OK);
}
void cellRtcConvertLocalTimeToUtc(ppu::Context& c) {
  K().w64(a32(c, 1), K().r64(a32(c, 0)) - uint64_t(local_offset_us()));
  ret(c, CELL_OK);
}

}  // namespace

void register_builtin_modules() {
  static bool done = false;
  if (done) return;
  done = true;
  register_fs_modules();
  register_pad_modules();
  register_gcm_modules();
  register_audio_modules();
  lv2::register_spu_modules();
  register_misc_modules();
  register_spurs_module();
  register_savedata_module();
#define R(mod, fn) register_function(mod, #fn, fn)
#define R_AS(mod, name, fn) register_function(mod, name, fn)
  const char* P = "sysPrxForUser";
  R(P, sys_initialize_tls);
  R(P, sys_process_exit);
  R_AS(P, "_sys_process_atexitspawn", sys_process_atexitspawn);
  R_AS(P, "_sys_process_at_Exitspawn", sys_process_atexitspawn);
  R(P, sys_process_is_stack);
  R(P, sys_prx_exitspawn_with_level);
  R(P, sys_ppu_thread_create);
  R(P, sys_ppu_thread_exit);
  R(P, sys_ppu_thread_get_id);
  R(P, sys_time_get_system_time);
  R(P, sys_lwmutex_create);
  R(P, sys_lwmutex_destroy);
  R(P, sys_lwmutex_lock);
  R(P, sys_lwmutex_trylock);
  R(P, sys_lwmutex_unlock);
  R(P, sys_lwcond_create);
  R(P, sys_lwcond_destroy);
  R(P, sys_lwcond_signal);
  R(P, sys_lwcond_signal_all);
  R(P, sys_lwcond_signal_to);
  R(P, sys_lwcond_wait);
  R(P, _sys_malloc);
  R(P, _sys_memalign);
  R(P, _sys_free);
  R(P, _sys_memset);
  R(P, _sys_memcpy);
  R(P, _sys_memmove);
  R(P, _sys_memcmp);
  R(P, _sys_memchr);
  R(P, _sys_strlen);
  R(P, _sys_strcmp);
  R(P, _sys_strncmp);
  R(P, _sys_strcpy);
  R(P, _sys_strncpy);
  R(P, _sys_strcat);
  R(P, _sys_strchr);
  R(P, _sys_strrchr);
  R(P, _sys_tolower);
  R(P, _sys_toupper);
  R(P, _sys_heap_create_heap);
  R(P, _sys_heap_delete_heap);
  R(P, _sys_heap_malloc);
  R(P, _sys_heap_memalign);
  R(P, _sys_heap_free);
  R(P, _sys_printf);
  R(P, _sys_sprintf);
  R(P, _sys_snprintf);

  const char* M = "cellSysmodule";
  R_AS(M, "cellSysmoduleInitialize", ok);
  R_AS(M, "cellSysmoduleFinalize", ok);
  R(M, cellSysmoduleLoadModule);
  R_AS(M, "cellSysmoduleUnloadModule", ok);
  R_AS(M, "cellSysmoduleIsLoaded", ok);
  R_AS(M, "cellSysmoduleSetMemcontainer", ok);

  const char* U = "cellSysutil";
  R(U, cellSysutilRegisterCallback);
  R(U, cellSysutilUnregisterCallback);
  R(U, cellSysutilCheckCallback);
  R(U, cellSysutilGetSystemParamInt);
  R(U, cellSysutilGetSystemParamString);
  R(U, cellMsgDialogOpen2);
  R(U, cellMsgDialogOpenErrorCode);
  for (const char* n : {"cellMsgDialogAbort", "cellMsgDialogClose", "cellMsgDialogProgressBarInc",
                        "cellMsgDialogProgressBarSetMsg", "cellMsgDialogProgressBarReset"})
    register_function(U, n, ok);

  R(P, sys_ppu_thread_once);
  R(P, sys_mmapper_allocate_memory);
  R(P, sys_mmapper_map_memory);
  R(P, sys_mmapper_free_memory);
  R(P, sys_mmapper_unmap_memory);
  R(P, sys_prx_load_module);
  R(P, sys_prx_start_module);
  R_AS(P, "sys_prx_stop_module", ok);
  R_AS(P, "sys_prx_unload_module", ok);
  R_AS(P, "sys_prx_register_library", ok);

  const char* T = "cellRtc";
  R(T, cellRtcGetCurrentTick);
  R(T, cellRtcGetCurrentClock);
  R(T, cellRtcGetCurrentClockLocalTime);
  R(T, cellRtcGetTick);
  R(T, cellRtcSetTick);
  R(T, cellRtcGetDayOfWeek);
  R(T, cellRtcConvertUtcToLocalTime);
  R(T, cellRtcConvertLocalTimeToUtc);
#undef R
#undef R_AS
}

}  // namespace hle
