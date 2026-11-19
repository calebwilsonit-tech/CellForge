// Small system services that games query during startup: system cache, network
// control, the NP manager and trophies. Everything here reports an offline
// console with no PSN account; outputs are always written so the game never reads
// uninitialised memory.

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

#include "hle.h"
#include "lv2.h"
#include "vfs.h"

namespace hle {
namespace {

using lv2::Kernel;
Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }
void ok(ppu::Context& c) { ret(c, 0); }
uint8_t* G(uint32_t a) { return K().base() + a; }

// ---- cellSysCache (in cellSysutil) ----------------------------------------------
// CellSysCacheParam: char cacheId[32]; char getCachePath[1024]; void* reserved.
std::string g_cache_path;
void cellSysCacheMount(ppu::Context& c) {
  const uint32_t p = a32(c, 0);
  const std::string id = K().read_str(p, 31);
  const std::string title = vfs::game_sfo().strings.count("TITLE_ID") ? vfs::game_sfo().strings.at("TITLE_ID") : "GAME";
  g_cache_path = "/dev_hdd1/caches/" + title + (id.empty() ? "" : "_" + id);
  bool existed = false;
  if (auto r = vfs::resolve(g_cache_path)) {
    std::error_code ec;
    existed = std::filesystem::exists(r->host, ec);
    std::filesystem::create_directories(r->host, ec);
  }
  std::memset(G(p + 32), 0, 1024);
  std::memcpy(G(p + 32), g_cache_path.c_str(), g_cache_path.size());
  K().logf("cellSysCacheMount: %s", g_cache_path.c_str());
  ret(c, existed ? 1 : 0);  // CELL_SYSCACHE_RET_OK_RELAYED / OK_CLEARED
}
void cellSysCacheClear(ppu::Context& c) {
  if (!g_cache_path.empty())
    if (auto r = vfs::resolve(g_cache_path)) {
      std::error_code ec;
      for (auto& e : std::filesystem::directory_iterator(r->host, ec)) std::filesystem::remove_all(e.path(), ec);
    }
  ok(c);
}

// ---- cellNetCtl: never connected ------------------------------------------------
constexpr uint32_t kNetCtlNotConnected = 0x80130108;
constexpr uint32_t kNetCtlDialogCanceled = 0x80130190;

// Network start dialog: no UI, the player "cancels" at once so the game carries on
// offline. Events: LOADED 0x0801, FINISHED 0x0802, UNLOADED 0x0803.
void cellNetCtlNetStartDialogLoadAsync(ppu::Context& c) {
  K().logf("network start dialog: answered as cancelled (offline)");
  queue_sysutil_event(0x0801, 0);
  queue_sysutil_event(0x0802, 0);
  ok(c);
}
void cellNetCtlNetStartDialogAbortAsync(ppu::Context& c) { ok(c); }
void cellNetCtlNetStartDialogUnloadAsync(ppu::Context& c) {  // (CellNetCtlNetStartDialogResult* {u32 size; s32 result})
  if (const uint32_t r = a32(c, 0)) K().w32(r + 4, kNetCtlDialogCanceled);
  queue_sysutil_event(0x0803, 0);
  ok(c);
}
void cellNetCtlGetState(ppu::Context& c) { K().w32(a32(c, 0), 0); ok(c); }  // Disconnected
void cellNetCtlGetInfo(ppu::Context& c) { ret(c, kNetCtlNotConnected); }
void cellNetCtlGetNatInfo(ppu::Context& c) {
  // CellNetCtlNatInfo: u32 size; s32 upnpStatus; s32 stunStatus; s32 natType; u32 mappedAddr
  const uint32_t p = a32(c, 0);
  if (p) { K().w32(p + 4, 0); K().w32(p + 8, 0); K().w32(p + 12, 0); K().w32(p + 16, 0); }
  ret(c, kNetCtlNotConnected);
}
void cellNetCtlAddHandler(ppu::Context& c) { K().w32(a32(c, 2), 1); ok(c); }  // (handler, arg, s32* hid)

// ---- cellOskDialog (in cellSysutil): no keyboard UI; text is confirmed at once ----
// Result text: the game's initial text, else PS3RECOMP_OSK_TEXT, else "Recomp".
// Events: LOADED 0x0502, INPUT_ENTERED 0x0505, FINISHED 0x0503, UNLOADED 0x0504.
std::u16string g_osk_text;
void cellOskDialogLoadAsync(ppu::Context& c) {  // (container, CellOskDialogParam*, CellOskDialogInputFieldInfo*)
  g_osk_text.clear();
  if (const uint32_t info = a32(c, 2)) {
    // CellOskDialogInputFieldInfo: u16* message, u16* init_text, u32 limit_length (UTF-16 big-endian)
    if (const uint32_t init = K().r32(info + 4))
      for (uint32_t i = 0; i < 512; ++i) {
        const uint16_t ch = K().r16(init + 2 * i);
        if (!ch) break;
        g_osk_text.push_back(char16_t(ch));
      }
  }
  if (g_osk_text.empty()) {
    const char* env = std::getenv("PS3RECOMP_OSK_TEXT");
    for (const char* p = env ? env : "Recomp"; *p; ++p) g_osk_text.push_back(char16_t(uint8_t(*p)));
  }
  K().logf("on-screen keyboard: answered with %zu character(s)", g_osk_text.size());
  queue_sysutil_event(0x0502, 0);
  queue_sysutil_event(0x0505, 0);
  queue_sysutil_event(0x0503, 0);
  ok(c);
}
// CellOskDialogCallbackReturnParam: s32 result, s32 numCharsResultString (capacity), u16* pResultString
void write_osk_result(uint32_t out) {
  if (!out) return;
  K().w32(out, 0);  // CELL_OSKDIALOG_INPUT_FIELD_RESULT_OK
  const uint32_t cap = K().r32(out + 4), str = K().r32(out + 8);
  if (!str || !cap) return;
  const uint32_t n = std::min<uint32_t>(uint32_t(g_osk_text.size()), cap - 1);
  for (uint32_t i = 0; i < n; ++i) K().w16(str + 2 * i, uint16_t(g_osk_text[i]));
  K().w16(str + 2 * n, 0);
}
void cellOskDialogUnloadAsync(ppu::Context& c) {
  write_osk_result(a32(c, 0));
  queue_sysutil_event(0x0504, 0);
  ok(c);
}
void cellOskDialogGetInputText(ppu::Context& c) { write_osk_result(a32(c, 0)); ok(c); }
void cellSysutilGetBgmPlaybackStatus(ppu::Context& c) {
  if (const uint32_t p = a32(c, 0)) std::memset(G(p), 0, 32);  // stopped, disabled
  ok(c);
}

// ---- sceNp: signed out ------------------------------------------------------------
void sceNpManagerGetStatus(ppu::Context& c) { K().w32(a32(c, 0), uint32_t(-1)); ok(c); }  // OFFLINE
void sceNpManagerGetChatRestrictionFlag(ppu::Context& c) { K().w32(a32(c, 0), 0); ok(c); }
void sceNpManagerGetContentRatingFlag(ppu::Context& c) { K().w32(a32(c, 0), 0); K().w32(a32(c, 1), 18); ok(c); }
void sceNpManagerGetAccountAge(ppu::Context& c) { K().w32(a32(c, 0), 18); ok(c); }
void sceNpManagerGetNpId(ppu::Context& c) {
  std::memset(G(a32(c, 0)), 0, 36);  // SceNpId
  ret(c, 0x8002AA0C);  // not signed in
}

// ---- sceNpTrophy: a local context with nothing unlocked ---------------------------
void sceNpTrophyCreateContext(ppu::Context& c) { K().w32(a32(c, 0), 1); ok(c); }
void sceNpTrophyCreateHandle(ppu::Context& c) { K().w32(a32(c, 0), 1); ok(c); }
void sceNpTrophyGetRequiredDiskSpace(ppu::Context& c) { K().w64(a32(c, 2), 0); ok(c); }
void sceNpTrophyGetTrophyUnlockState(ppu::Context& c) {
  std::memset(G(a32(c, 2)), 0, 16);  // SceNpTrophyFlagArray (128 bits)
  K().w32(a32(c, 3), 0);
  ok(c);
}
void sceNpTrophyUnlockTrophy(ppu::Context& c) {
  K().logf("trophy %d unlocked", int32_t(c.r[5]));
  K().w32(a32(c, 3), uint32_t(-1));  // no platinum
  ok(c);
}

}  // namespace

void register_misc_modules() {
#define R(mod, fn) register_function(mod, #fn, fn)
  R("cellSysutil", cellSysCacheMount);
  R("cellSysutil", cellSysCacheClear);
  R("cellSysutil", cellOskDialogLoadAsync);
  R("cellSysutil", cellOskDialogUnloadAsync);
  R("cellSysutil", cellOskDialogGetInputText);
  R("cellSysutil", cellSysutilGetBgmPlaybackStatus);
  for (const char* n : {"cellOskDialogSetKeyLayoutOption", "cellOskDialogSetLayoutMode", "cellOskDialogSetDeviceMask",
                        "cellOskDialogAddSupportLanguage", "cellOskDialogSetInitialInputDevice", "cellOskDialogAbort",
                        "cellOskDialogDisableDimmer", "cellOskDialogSetSeparateWindowOption", "cellOskDialogSetInitialKeyLayout"})
    register_function("cellSysutil", n, ok);
  R("cellNetCtl", cellNetCtlGetState);
  R("cellNetCtl", cellNetCtlGetInfo);
  R("cellNetCtl", cellNetCtlGetNatInfo);
  R("cellNetCtl", cellNetCtlAddHandler);
  R("cellNetCtl", cellNetCtlNetStartDialogLoadAsync);
  R("cellNetCtl", cellNetCtlNetStartDialogAbortAsync);
  R("cellNetCtl", cellNetCtlNetStartDialogUnloadAsync);
  for (const char* n : {"cellNetCtlInit", "cellNetCtlTerm", "cellNetCtlDelHandler"}) register_function("cellNetCtl", n, ok);
  R("sceNp", sceNpManagerGetStatus);
  R("sceNp", sceNpManagerGetChatRestrictionFlag);
  R("sceNp", sceNpManagerGetContentRatingFlag);
  R("sceNp", sceNpManagerGetAccountAge);
  R("sceNp", sceNpManagerGetNpId);
  for (const char* n : {"sceNpInit", "sceNpTerm", "sceNpDrmIsAvailable", "sceNpManagerRegisterCallback",
                        "sceNpManagerUnregisterCallback", "sceNpBasicUnregisterHandler"})
    register_function("sceNp", n, ok);
  R("sceNpTrophy", sceNpTrophyCreateContext);
  R("sceNpTrophy", sceNpTrophyCreateHandle);
  R("sceNpTrophy", sceNpTrophyGetRequiredDiskSpace);
  R("sceNpTrophy", sceNpTrophyGetTrophyUnlockState);
  R("sceNpTrophy", sceNpTrophyUnlockTrophy);
  for (const char* n : {"sceNpTrophyInit", "sceNpTrophyRegisterContext"}) register_function("sceNpTrophy", n, ok);
#undef R
}

}  // namespace hle
