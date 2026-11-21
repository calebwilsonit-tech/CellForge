// Small system services that games query during startup: system cache, network
// control, the NP manager and trophies. Everything here reports an offline
// console with no PSN account; outputs are always written so the game never reads
// uninitialised memory.

#define _CRT_SECURE_NO_WARNINGS
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

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

// ---- cellOskDialog (in cellSysutil): a real text-entry window --------------------
// The PS3 on-screen keyboard is replaced by a small native dialog (type with the PC keyboard, Enter = OK,
// Esc = cancel). Events: LOADED 0x0502, FINISHED 0x0503 (param: 0 confirmed, 1 cancelled, 2 aborted),
// UNLOADED 0x0504, INPUT_ENTERED 0x0505. PS3RECOMP_OSK_TEXT answers instantly without a window (automated runs).
std::u16string g_osk_text;
std::atomic<int> g_osk_result{0};  // CellOskDialogInputFieldResult: 0 OK, 1 canceled, 2 aborted, 3 no text
std::atomic<bool> g_osk_open{false};
std::atomic<void*> g_osk_hwnd{nullptr};

std::u16string read_guest_u16(uint32_t addr, uint32_t max_chars) {
  std::u16string s;
  if (!addr) return s;
  for (uint32_t i = 0; i < max_chars; ++i) {
    const uint16_t ch = K().r16(addr + 2 * i);
    if (!ch) break;
    s.push_back(char16_t(ch));
  }
  return s;
}

void osk_finish(int result, const std::u16string& text) {
  g_osk_text = text;
  g_osk_result = result;
  if (result == 0) queue_sysutil_event(0x0505, 0);
  queue_sysutil_event(0x0503, result == 0 ? 0 : result);
  g_osk_open = false;
}

#if defined(_WIN32)
struct OskDlg {
  std::wstring title, init;
  uint32_t limit = 0;
  HWND edit = nullptr;
  int result = 1;
  std::u16string text;
  bool done = false;
};

LRESULT CALLBACK osk_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
  auto* d = reinterpret_cast<OskDlg*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  switch (m) {
    case WM_COMMAND:
      if (d && LOWORD(w) == IDOK) {
        const int n = GetWindowTextLengthW(d->edit);
        std::wstring t(size_t(n) + 1, L'\0');
        GetWindowTextW(d->edit, t.data(), n + 1);
        t.resize(size_t(n));
        d->text.assign(t.begin(), t.end());
        d->result = d->text.empty() ? 3 : 0;
        d->done = true;
        return 0;
      }
      if (d && LOWORD(w) == IDCANCEL) { d->result = 1; d->done = true; return 0; }
      break;
    case WM_CLOSE:
      if (d) { d->result = g_osk_result.load() == 2 ? 2 : 1; d->done = true; }
      return 0;
  }
  return DefWindowProcW(h, m, w, l);
}

void osk_thread(OskDlg dlg, HWND owner) {
  static const wchar_t* kClass = L"CellForgeOskDialog";
  static const bool registered = [] {
    WNDCLASSW wc{};
    wc.lpfnWndProc = osk_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wc.hbrBackground = reinterpret_cast<HBRUSH>(static_cast<INT_PTR>(COLOR_BTNFACE + 1));
    wc.lpszClassName = kClass;
    return RegisterClassW(&wc) != 0;
  }();
  (void)registered;
  NONCLIENTMETRICSW ncm{};
  ncm.cbSize = sizeof ncm;
  SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof ncm, &ncm, 0);
  ncm.lfMessageFont.lfHeight = ncm.lfMessageFont.lfHeight * 5 / 4;
  HFONT font = CreateFontIndirectW(&ncm.lfMessageFont);

  const int W = 460, H = 190;
  RECT r{};
  if (!(owner && GetWindowRect(owner, &r))) r = RECT{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
  const int x = r.left + ((r.right - r.left) - W) / 2, y = r.top + ((r.bottom - r.top) - H) / 2;
  HWND h = CreateWindowExW(WS_EX_TOPMOST | WS_EX_DLGMODALFRAME | WS_EX_CONTROLPARENT, kClass, dlg.title.c_str(),
                           WS_POPUP | WS_CAPTION | WS_SYSMENU, x, y, W, H, owner, nullptr, GetModuleHandleW(nullptr), nullptr);
  if (!h) { osk_finish(1, u""); return; }
  SetWindowLongPtrW(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&dlg));
  auto mk = [&](const wchar_t* cls, const wchar_t* text, DWORD style, int cx, int cy, int cw, int ch, int id, DWORD ex = 0) {
    HWND c = CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, cx, cy, cw, ch, h,
                             reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), GetModuleHandleW(nullptr), nullptr);
    SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    return c;
  };
  mk(L"STATIC", dlg.title.c_str(), SS_LEFT, 16, 14, W - 48, 24, 100);
  dlg.edit = mk(L"EDIT", dlg.init.c_str(), WS_TABSTOP | ES_AUTOHSCROLL, 16, 48, W - 48, 28, 101, WS_EX_CLIENTEDGE);
  if (dlg.limit) SendMessageW(dlg.edit, EM_LIMITTEXT, dlg.limit, 0);
  mk(L"BUTTON", L"OK", WS_TABSTOP | BS_DEFPUSHBUTTON, W - 48 - 176, 96, 80, 30, IDOK);
  mk(L"BUTTON", L"Cancel", WS_TABSTOP | BS_PUSHBUTTON, W - 48 - 88, 96, 80, 30, IDCANCEL);
  ShowWindow(h, SW_SHOW);
  SetForegroundWindow(h);
  SetFocus(dlg.edit);
  SendMessageW(dlg.edit, EM_SETSEL, 0, -1);
  g_osk_hwnd = h;
  MSG msg;
  while (!dlg.done && GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(h, &msg)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
  }
  g_osk_hwnd = nullptr;
  DestroyWindow(h);
  DeleteObject(font);
  if (owner) SetForegroundWindow(owner);
  osk_finish(dlg.result, dlg.text);
}
#endif

void cellOskDialogLoadAsync(ppu::Context& c) {  // (container, CellOskDialogParam*, CellOskDialogInputFieldInfo*)
  // CellOskDialogInputFieldInfo: u16* message, u16* init_text, u32 limit_length (UTF-16 big-endian)
  std::u16string message, init;
  uint32_t limit = 0;
  if (const uint32_t info = a32(c, 2)) {
    message = read_guest_u16(K().r32(info), 128);
    init = read_guest_u16(K().r32(info + 4), 512);
    limit = K().r32(info + 8);
  }
  queue_sysutil_event(0x0502, 0);
  if (g_osk_open.exchange(true)) { ok(c); return; }
  if (const char* env = std::getenv("PS3RECOMP_OSK_TEXT")) {  // automated runs: no window
    std::u16string t;
    for (const char* p = env; *p; ++p) t.push_back(char16_t(uint8_t(*p)));
    K().logf("on-screen keyboard: answered with PS3RECOMP_OSK_TEXT (%zu character(s))", t.size());
    osk_finish(0, t);
    ok(c);
    return;
  }
#if defined(_WIN32)
  OskDlg dlg;
  dlg.title = message.empty() ? L"Enter text" : std::wstring(message.begin(), message.end());
  dlg.init.assign(init.begin(), init.end());
  dlg.limit = limit;
  K().logf("on-screen keyboard: showing a text entry window (limit %u)", limit);
  std::thread(osk_thread, std::move(dlg), GetForegroundWindow()).detach();
#else
  osk_finish(0, init.empty() ? u"Recomp" : init);
#endif
  ok(c);
}
// CellOskDialogCallbackReturnParam: s32 result, s32 numCharsResultString (capacity), u16* pResultString
void write_osk_result(uint32_t out) {
  if (!out) return;
  K().w32(out, uint32_t(g_osk_result.load()));
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
void cellOskDialogAbort(ppu::Context& c) {
#if defined(_WIN32)
  if (void* h = g_osk_hwnd.load()) { g_osk_result = 2; PostMessageW(static_cast<HWND>(h), WM_CLOSE, 0, 0); }
#endif
  ok(c);
}
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
  R("cellSysutil", cellOskDialogAbort);
  R("cellSysutil", cellSysutilGetBgmPlaybackStatus);
  for (const char* n : {"cellOskDialogSetKeyLayoutOption", "cellOskDialogSetLayoutMode", "cellOskDialogSetDeviceMask",
                        "cellOskDialogAddSupportLanguage", "cellOskDialogSetInitialInputDevice",
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
