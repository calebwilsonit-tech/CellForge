#include "settings_ui.h"
#include "options.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>

namespace cfg {
namespace {

enum : int {
  ID_PRESET = 201,
  ID_DISPLAY_MODE,
  ID_RES,
  ID_VSYNC,
  ID_FPS_CAP,
  ID_AF,
  ID_UPSCALE,
  ID_MIPMAPS,
  ID_FPS_COUNTER,
  ID_VOLUME,
  ID_MUTE_BG,
  ID_APPLY,
  ID_RESET,
  ID_CLOSE,
};

constexpr COLORREF kBg = RGB(18, 21, 28);
constexpr COLORREF kCard = RGB(27, 32, 43);
constexpr COLORREF kText = RGB(229, 233, 240);
constexpr COLORREF kMuted = RGB(148, 163, 184);
constexpr COLORREF kAccent = RGB(37, 99, 235);
constexpr COLORREF kGreen = RGB(16, 185, 129);
constexpr COLORREF kSlate = RGB(51, 65, 85);

struct ResEntry {
  uint32_t w, h;
  const wchar_t* label;
};
constexpr ResEntry kResolutions[] = {
    {1280, 720, L"1280 x 720  (720p Native)"},
    {1600, 900, L"1600 x 900  (900p)"},
    {1920, 1080, L"1920 x 1080 (1080p Full HD)"},
    {2560, 1440, L"2560 x 1440 (1440p QHD)"},
    {3840, 2160, L"3840 x 2160 (4K UHD)"},
};
constexpr int kAfVals[] = {1, 2, 4, 8, 16};
constexpr int kFpsVals[] = {30, 60, 120, 0};
constexpr int kVolVals[] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

struct DialogState {
  std::filesystem::path ini;
  Options opt;
  bool saved = false;
  bool updating = false;
  HWND hwnd = nullptr;
  HWND c_preset = nullptr, c_disp = nullptr, c_res = nullptr, c_vsync = nullptr, c_fps = nullptr;
  HWND c_af = nullptr, c_upscale = nullptr, c_mips = nullptr, c_fps_cnt = nullptr;
  HWND c_vol = nullptr, c_mute = nullptr, c_hint = nullptr;
  HFONT f_ui = nullptr, f_bold = nullptr, f_hdr = nullptr;
  HBRUSH br_bg = nullptr, br_card = nullptr;
};

void enable_dark_bar(HWND hwnd) {
  if (HMODULE dwm = LoadLibraryW(L"dwmapi.dll")) {
    using Fn = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
    if (auto fn = reinterpret_cast<Fn>(GetProcAddress(dwm, "DwmSetWindowAttribute"))) {
      BOOL dark = TRUE;
      if (FAILED(fn(hwnd, 20, &dark, sizeof(dark)))) fn(hwnd, 19, &dark, sizeof(dark));
    }
    FreeLibrary(dwm);
  }
}

void push_to_controls(DialogState& s) {
  s.updating = true;
  SendMessageW(s.c_preset, CB_SETCURSEL, std::clamp(s.opt.quality_preset, 0, 4), 0);
  SendMessageW(s.c_disp, CB_SETCURSEL, std::clamp(s.opt.display_mode, 0, 1), 0);

  int res_idx = 0;
  for (int i = 0; i < int(std::size(kResolutions)); ++i)
    if (kResolutions[i].w == s.opt.window_width && kResolutions[i].h == s.opt.window_height) res_idx = i;
  SendMessageW(s.c_res, CB_SETCURSEL, res_idx, 0);

  SendMessageW(s.c_vsync, CB_SETCURSEL, std::clamp(s.opt.vsync, 0, 2), 0);

  int fps_idx = 1;
  for (int i = 0; i < int(std::size(kFpsVals)); ++i)
    if (kFpsVals[i] == s.opt.fps_cap) fps_idx = i;
  SendMessageW(s.c_fps, CB_SETCURSEL, fps_idx, 0);

  int af_idx = 4;
  for (int i = 0; i < int(std::size(kAfVals)); ++i)
    if (kAfVals[i] == s.opt.anisotropic_filtering) af_idx = i;
  SendMessageW(s.c_af, CB_SETCURSEL, af_idx, 0);

  SendMessageW(s.c_upscale, CB_SETCURSEL, std::clamp(s.opt.upscale_filter, 0, 1), 0);
  SendMessageW(s.c_mips, BM_SETCHECK, s.opt.mipmaps ? BST_CHECKED : BST_UNCHECKED, 0);
  SendMessageW(s.c_fps_cnt, BM_SETCHECK, s.opt.fps_counter ? BST_CHECKED : BST_UNCHECKED, 0);

  int vol_idx = std::clamp((s.opt.volume + 5) / 10, 0, 10);
  SendMessageW(s.c_vol, CB_SETCURSEL, vol_idx, 0);
  SendMessageW(s.c_mute, BM_SETCHECK, s.opt.mute_in_background ? BST_CHECKED : BST_UNCHECKED, 0);
  s.updating = false;
}

void pull_from_controls(DialogState& s) {
  s.opt.quality_preset = int(SendMessageW(s.c_preset, CB_GETCURSEL, 0, 0));
  s.opt.display_mode = std::clamp(int(SendMessageW(s.c_disp, CB_GETCURSEL, 0, 0)), 0, 1);
  const int ri = std::clamp(int(SendMessageW(s.c_res, CB_GETCURSEL, 0, 0)), 0, int(std::size(kResolutions)) - 1);
  s.opt.window_width = kResolutions[ri].w;
  s.opt.window_height = kResolutions[ri].h;
  s.opt.vsync = std::clamp(int(SendMessageW(s.c_vsync, CB_GETCURSEL, 0, 0)), 0, 2);
  const int fi = std::clamp(int(SendMessageW(s.c_fps, CB_GETCURSEL, 0, 0)), 0, int(std::size(kFpsVals)) - 1);
  s.opt.fps_cap = kFpsVals[fi];
  const int ai = std::clamp(int(SendMessageW(s.c_af, CB_GETCURSEL, 0, 0)), 0, int(std::size(kAfVals)) - 1);
  s.opt.anisotropic_filtering = kAfVals[ai];
  s.opt.upscale_filter = std::clamp(int(SendMessageW(s.c_upscale, CB_GETCURSEL, 0, 0)), 0, 1);
  s.opt.mipmaps = SendMessageW(s.c_mips, BM_GETCHECK, 0, 0) == BST_CHECKED;
  s.opt.fps_counter = SendMessageW(s.c_fps_cnt, BM_GETCHECK, 0, 0) == BST_CHECKED;
  const int vi = std::clamp(int(SendMessageW(s.c_vol, CB_GETCURSEL, 0, 0)), 0, int(std::size(kVolVals)) - 1);
  s.opt.volume = kVolVals[vi];
  s.opt.mute_in_background = SendMessageW(s.c_mute, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

HWND mk(HWND parent, HFONT font, const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int h, int id = 0) {
  HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, parent, HMENU(INT_PTR(id)),
                           GetModuleHandleW(nullptr), nullptr);
  SendMessageW(c, WM_SETFONT, WPARAM(font), TRUE);
  return c;
}

void draw_btn(const DialogState& s, const DRAWITEMSTRUCT* dis) {
  COLORREF col = kSlate;
  if (dis->CtlID == ID_APPLY) col = kGreen;
  else if (dis->CtlID == ID_RESET) col = kAccent;
  if (dis->itemState & ODS_SELECTED)
    col = RGB(std::max(0, int(GetRValue(col)) - 25), std::max(0, int(GetGValue(col)) - 25),
              std::max(0, int(GetBValue(col)) - 25));
  HBRUSH br = CreateSolidBrush(col);
  FillRect(dis->hDC, &dis->rcItem, br);
  DeleteObject(br);
  SetBkMode(dis->hDC, TRANSPARENT);
  SetTextColor(dis->hDC, RGB(255, 255, 255));
  SelectObject(dis->hDC, s.f_bold);
  wchar_t text[64] = {};
  GetWindowTextW(dis->hwndItem, text, 64);
  RECT r = dis->rcItem;
  DrawTextW(dis->hDC, text, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

LRESULT CALLBACK dlg_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  auto* s = reinterpret_cast<DialogState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
  switch (msg) {
    case WM_CREATE: {
      auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
      s = reinterpret_cast<DialogState*>(cs->lpCreateParams);
      s->hwnd = hwnd;
      SetWindowLongPtrW(hwnd, GWLP_USERDATA, LONG_PTR(s));
      enable_dark_bar(hwnd);

      s->f_ui = CreateFontW(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      s->f_bold = CreateFontW(-14, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      s->f_hdr = CreateFontW(-16, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      s->br_bg = CreateSolidBrush(kBg);
      s->br_card = CreateSolidBrush(kCard);

      const int lx = 20, lw = 195, cx = 225, cw = 250;
      int y = 16;
      mk(hwnd, s->f_hdr, L"STATIC", L"Display && Graphics Settings", 0, lx, y, 440, 24);
      y += 34;

      auto add_combo = [&](const wchar_t* label, int id) -> HWND {
        mk(hwnd, s->f_ui, L"STATIC", label, 0, lx, y + 4, lw, 20);
        HWND cb = mk(hwnd, s->f_ui, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, cx, y, cw, 220, id);
        y += 34;
        return cb;
      };

      s->c_preset = add_combo(L"Quality Preset:", ID_PRESET);
      for (auto* p : {L"Low / Handheld (Fastest)", L"Balanced (720p, 4x AF)", L"High (1080p, 16x AF)",
                      L"Ultra (1440p, 16x AF)", L"Custom"})
        SendMessageW(s->c_preset, CB_ADDSTRING, 0, LPARAM(p));

      s->c_disp = add_combo(L"Display Mode:", ID_DISPLAY_MODE);
      SendMessageW(s->c_disp, CB_ADDSTRING, 0, LPARAM(L"Windowed"));
      SendMessageW(s->c_disp, CB_ADDSTRING, 0, LPARAM(L"Borderless Fullscreen (Alt+Enter)"));

      s->c_res = add_combo(L"Window Resolution:", ID_RES);
      for (const auto& r : kResolutions) SendMessageW(s->c_res, CB_ADDSTRING, 0, LPARAM(r.label));

      s->c_vsync = add_combo(L"V-Sync:", ID_VSYNC);
      SendMessageW(s->c_vsync, CB_ADDSTRING, 0, LPARAM(L"Off (Lowest Latency)"));
      SendMessageW(s->c_vsync, CB_ADDSTRING, 0, LPARAM(L"Fast / Mailbox (Tear-Free)"));
      SendMessageW(s->c_vsync, CB_ADDSTRING, 0, LPARAM(L"On / FIFO (Standard V-Sync)"));

      s->c_fps = add_combo(L"Frame Rate Cap:", ID_FPS_CAP);
      for (auto* f : {L"30 FPS", L"60 FPS (Native)", L"120 FPS", L"Uncapped"})
        SendMessageW(s->c_fps, CB_ADDSTRING, 0, LPARAM(f));

      s->c_af = add_combo(L"Anisotropic Filtering:", ID_AF);
      for (auto* a : {L"Off (1x)", L"2x", L"4x", L"8x", L"16x (Recommended)"})
        SendMessageW(s->c_af, CB_ADDSTRING, 0, LPARAM(a));

      s->c_upscale = add_combo(L"Upscale Filter:", ID_UPSCALE);
      SendMessageW(s->c_upscale, CB_ADDSTRING, 0, LPARAM(L"Nearest (Crisp Pixels)"));
      SendMessageW(s->c_upscale, CB_ADDSTRING, 0, LPARAM(L"Linear (Smooth)"));

      s->c_mips = mk(hwnd, s->f_ui, L"BUTTON", L"Enable GPU Mipmaps && Trilinear Filtering (eliminates ground grain)",
                     BS_AUTOCHECKBOX | WS_TABSTOP, lx, y, 455, 22, ID_MIPMAPS);
      y += 28;
      s->c_fps_cnt = mk(hwnd, s->f_ui, L"BUTTON", L"Show live FPS counter in window title bar",
                        BS_AUTOCHECKBOX | WS_TABSTOP, lx, y, 455, 22, ID_FPS_COUNTER);
      y += 34;

      mk(hwnd, s->f_hdr, L"STATIC", L"Audio", 0, lx, y, 440, 22);
      y += 30;
      s->c_vol = add_combo(L"Master Volume:", ID_VOLUME);
      for (int v : kVolVals) {
        std::wstring lbl = std::to_wstring(v) + L"%";
        SendMessageW(s->c_vol, CB_ADDSTRING, 0, LPARAM(lbl.c_str()));
      }
      s->c_mute = mk(hwnd, s->f_ui, L"BUTTON", L"Mute audio when window is in background",
                     BS_AUTOCHECKBOX | WS_TABSTOP, lx, y, 455, 22, ID_MUTE_BG);
      y += 32;

      s->c_hint = mk(hwnd, s->f_ui, L"STATIC", L"Tip: Press F1 in-game to open Settings, Alt+Enter to toggle Fullscreen.",
                     0, lx, y, 455, 20);
      y += 30;

      mk(hwnd, s->f_bold, L"BUTTON", L"Save && Apply", BS_OWNERDRAW | WS_TABSTOP, lx, y, 145, 34, ID_APPLY);
      mk(hwnd, s->f_bold, L"BUTTON", L"Reset Defaults", BS_OWNERDRAW | WS_TABSTOP, lx + 155, y, 135, 34, ID_RESET);
      mk(hwnd, s->f_bold, L"BUTTON", L"Close", BS_OWNERDRAW | WS_TABSTOP, lx + 300, y, 155, 34, ID_CLOSE);

      push_to_controls(*s);
      return 0;
    }
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLORBTN: {
      if (!s) break;
      HDC hdc = HDC(wp);
      SetBkColor(hdc, kBg);
      SetTextColor(hdc, HWND(lp) == s->c_hint ? kMuted : kText);
      return LRESULT(s->br_bg);
    }
    case WM_CTLCOLORLISTBOX:
    case WM_CTLCOLOREDIT: {
      if (!s) break;
      HDC hdc = HDC(wp);
      SetBkColor(hdc, kCard);
      SetTextColor(hdc, kText);
      return LRESULT(s->br_card);
    }
    case WM_DRAWITEM:
      if (s) draw_btn(*s, reinterpret_cast<const DRAWITEMSTRUCT*>(lp));
      return TRUE;
    case WM_COMMAND: {
      if (!s) break;
      const int id = LOWORD(wp);
      const int code = HIWORD(wp);
      if (id == ID_PRESET && code == CBN_SELCHANGE && !s->updating) {
        const int p = int(SendMessageW(s->c_preset, CB_GETCURSEL, 0, 0));
        if (p >= 0 && p <= 3) {
          s->opt.apply_preset(p);
          push_to_controls(*s);
        }
      } else if ((id == ID_RES || id == ID_VSYNC || id == ID_FPS_CAP || id == ID_AF || id == ID_UPSCALE || id == ID_MIPMAPS) &&
                 (code == CBN_SELCHANGE || code == BN_CLICKED) && !s->updating) {
        SendMessageW(s->c_preset, CB_SETCURSEL, 4 /*Custom*/, 0);
      } else if (id == ID_APPLY) {
        pull_from_controls(*s);
        cfg::get() = s->opt;
        cfg::sync_atomics(s->opt);
        if (!s->ini.empty()) cfg::save(s->ini, s->opt);
        s->saved = true;
        SetWindowTextW(s->c_hint, L"Settings saved and applied!");
      } else if (id == ID_RESET) {
        s->opt = Options{};
        s->opt.apply_preset(2);
        push_to_controls(*s);
        SetWindowTextW(s->c_hint, L"Defaults restored — click Save & Apply to confirm.");
      } else if (id == ID_CLOSE) {
        DestroyWindow(hwnd);
      }
      return 0;
    }
    case WM_KEYDOWN:
      if (wp == VK_ESCAPE || wp == VK_F1) { DestroyWindow(hwnd); return 0; }
      break;
    case WM_DESTROY:
      if (s) {
        DeleteObject(s->f_ui);
        DeleteObject(s->f_bold);
        DeleteObject(s->f_hdr);
        DeleteObject(s->br_bg);
        DeleteObject(s->br_card);
      }
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

std::atomic<bool> g_async_open{false};

}  // namespace

bool show_settings_dialog(const std::filesystem::path& options_ini, void* owner_hwnd) {
  DialogState st;
  st.ini = options_ini;
  st.opt = cfg::get();
  if (!options_ini.empty()) cfg::load(options_ini, st.opt);

  HINSTANCE inst = GetModuleHandleW(nullptr);
  constexpr wchar_t kCls[] = L"CellForgeSettingsDlg";
  WNDCLASSEXW wc{sizeof(wc)};
  wc.lpfnWndProc = dlg_proc;
  wc.hInstance = inst;
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  wc.hbrBackground = CreateSolidBrush(kBg);
  wc.lpszClassName = kCls;
  RegisterClassExW(&wc);

  const int W = 515, H = 600;
  RECT area{};
  SystemParametersInfoW(SPI_GETWORKAREA, 0, &area, 0);
  const int x = (area.left + area.right - W) / 2;
  const int y = (area.top + area.bottom - H) / 2;
  HWND hwnd = CreateWindowExW(WS_EX_DLGMODALFRAME, kCls, L"CellForge — Graphics & Audio Settings",
                              WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
                              x, y, W, H, static_cast<HWND>(owner_hwnd), nullptr, inst, &st);
  if (!hwnd) return false;
  ShowWindow(hwnd, SW_SHOW);
  UpdateWindow(hwnd);
  SetForegroundWindow(hwnd);

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(hwnd, &msg)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }
  return st.saved;
}

void toggle_settings_dialog_async(const std::filesystem::path& options_ini, void* owner_hwnd) {
  bool expected = false;
  if (!g_async_open.compare_exchange_strong(expected, true)) return;
  std::thread([options_ini, owner_hwnd] {
    show_settings_dialog(options_ini, owner_hwnd);
    g_async_open.store(false);
  }).detach();
}

}  // namespace cfg

#else

namespace cfg {
bool show_settings_dialog(const std::filesystem::path&, void*) { return false; }
void toggle_settings_dialog_async(const std::filesystem::path&, void*) {}
}  // namespace cfg

#endif
