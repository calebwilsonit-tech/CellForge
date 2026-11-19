// Skate3-VK Clip Studio — Instant PNG Screenshot & Trick-Sequence Burst Mod
// Hotkeys: F12 = Screenshot, F11 = Toggle Burst Mode, F5 = Open Clip Studio Panel.

#include "cellforge_mod.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

namespace {

struct StudioConfig {
  std::filesystem::path out_dir;
  int burst_fps = 5;
  bool watermark = true;
  bool shutter_beep = true;
};

StudioConfig g_cfg;
std::atomic<bool> g_burst_active{false};
std::atomic<int> g_shot_count{0};
std::atomic<bool> g_panel_open{false};
std::mutex g_last_mu;
std::string g_last_file;
std::filesystem::path g_burst_dir;

std::string expand_env(const std::string& s) {
  if (s.find('%') == std::string::npos) return s;
  char buf[4096];
  const DWORD n = ExpandEnvironmentStringsA(s.c_str(), buf, sizeof(buf));
  if (n > 0 && n < sizeof(buf)) return std::string(buf);
  return s;
}

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
  std::wstring out(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), out.data(), n);
  return out;
}

std::string timestamp_str(bool with_ms) {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
  const std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_s(&tm, &tt);
  char buf[64];
  if (with_ms) {
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d_%03d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, int(ms.count()));
  } else {
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
  }
  return buf;
}

bool ensure_gdiplus() {
  static const bool ok = [] {
    Gdiplus::GdiplusStartupInput in{};
    ULONG_PTR token = 0;
    return Gdiplus::GdiplusStartup(&token, &in, nullptr) == Gdiplus::Ok;
  }();
  return ok;
}

bool get_png_encoder(CLSID& clsid) {
  UINT num = 0, size = 0;
  Gdiplus::GetImageEncodersSize(&num, &size);
  if (size == 0) return false;
  std::vector<uint8_t> buf(size);
  auto* codecs = reinterpret_cast<Gdiplus::ImageCodecInfo*>(buf.data());
  if (Gdiplus::GetImageEncoders(num, size, codecs) != Gdiplus::Ok) return false;
  for (UINT i = 0; i < num; ++i) {
    if (wcscmp(codecs[i].MimeType, L"image/png") == 0) {
      clsid = codecs[i].Clsid;
      return true;
    }
  }
  return false;
}

// Find the main game window belonging to this process (largest visible window).
HWND find_game_window() {
  struct Search {
    DWORD pid;
    HWND best = nullptr;
    LONG best_area = 0;
  } s{GetCurrentProcessId()};

  EnumWindows(
      [](HWND h, LPARAM lp) -> BOOL {
        auto* sr = reinterpret_cast<Search*>(lp);
        DWORD pid = 0;
        GetWindowThreadProcessId(h, &pid);
        if (pid != sr->pid || !IsWindowVisible(h) || IsIconic(h)) return TRUE;
        RECT r{};
        if (GetClientRect(h, &r)) {
          const LONG area = (r.right - r.left) * (r.bottom - r.top);
          if (area > sr->best_area) {
            sr->best_area = area;
            sr->best = h;
          }
        }
        return TRUE;
      },
      LPARAM(&s));
  return s.best ? s.best : GetForegroundWindow();
}

bool capture_png(const std::filesystem::path& dest) {
  if (!ensure_gdiplus()) return false;
  HWND target = find_game_window();
  if (!target) return false;

  RECT rc{};
  if (!GetClientRect(target, &rc)) return false;
  const int w = rc.right - rc.left;
  const int h = rc.bottom - rc.top;
  if (w < 16 || h < 16) return false;

  POINT tl{0, 0};
  ClientToScreen(target, &tl);

  HDC screen = GetDC(nullptr);
  HDC mem = CreateCompatibleDC(screen);
  HBITMAP bmp = CreateCompatibleBitmap(screen, w, h);
  HGDIOBJ old = SelectObject(mem, bmp);
  BitBlt(mem, 0, 0, w, h, screen, tl.x, tl.y, SRCCOPY);

  if (g_cfg.watermark) {
    HFONT f = CreateFontW(-16, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
    HGDIOBJ old_f = SelectObject(mem, f);
    SetBkMode(mem, TRANSPARENT);
    RECT shadow{w - 139, h - 29, w - 9, h - 7};
    SetTextColor(mem, RGB(0, 0, 0));
    DrawTextW(mem, L"Skate3-VK", -1, &shadow, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    RECT fg{w - 140, h - 30, w - 10, h - 8};
    SetTextColor(mem, RGB(240, 244, 255));
    DrawTextW(mem, L"Skate3-VK", -1, &fg, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    SelectObject(mem, old_f);
    DeleteObject(f);
  }

  SelectObject(mem, old);
  DeleteDC(mem);
  ReleaseDC(nullptr, screen);

  std::error_code ec;
  std::filesystem::create_directories(dest.parent_path(), ec);

  CLSID png_clsid{};
  bool saved = false;
  if (get_png_encoder(png_clsid)) {
    Gdiplus::Bitmap gdip_bmp(bmp, nullptr);
    saved = (gdip_bmp.Save(dest.c_str(), &png_clsid, nullptr) == Gdiplus::Ok);
  }
  DeleteObject(bmp);

  if (saved) {
    g_shot_count.fetch_add(1);
    {
      std::lock_guard<std::mutex> lk(g_last_mu);
      g_last_file = dest.filename().string();
    }
    cellforge::logf("skate3_clip_studio: saved %s", dest.string().c_str());
  }
  return saved;
}

void take_single_shot() {
  const auto dest = g_cfg.out_dir / ("Skate3-VK_" + timestamp_str(true) + ".png");
  if (capture_png(dest) && g_cfg.shutter_beep) {
    Beep(1200, 45);
  }
}

void toggle_burst_mode() {
  const bool next = !g_burst_active.load();
  if (next) {
    g_burst_dir = g_cfg.out_dir / ("Burst_" + timestamp_str(false));
    g_burst_active.store(true);
    cellforge::logf("skate3_clip_studio: burst mode ON (%d FPS) -> %s", g_cfg.burst_fps, g_burst_dir.string().c_str());
    if (g_cfg.shutter_beep) Beep(900, 60);
  } else {
    g_burst_active.store(false);
    cellforge::log("skate3_clip_studio: burst mode OFF");
    if (g_cfg.shutter_beep) Beep(600, 60);
  }
}

// ---- F5 Dark-Mode Clip Studio Panel ----------------------------------------

constexpr COLORREF kBg = RGB(18, 21, 28);
constexpr COLORREF kText = RGB(229, 233, 240);
constexpr COLORREF kMuted = RGB(148, 163, 184);
constexpr COLORREF kAccent = RGB(37, 99, 235);
constexpr COLORREF kGreen = RGB(16, 185, 129);
constexpr COLORREF kAmber = RGB(217, 119, 6);
constexpr COLORREF kSlate = RGB(51, 65, 85);

enum : int { ID_SHOT = 401, ID_BURST, ID_OPEN_DIR, ID_CLOSE };

struct PanelState {
  HWND c_status = nullptr, c_last = nullptr;
  HFONT f_ui = nullptr, f_bold = nullptr, f_hdr = nullptr;
  HBRUSH br_bg = nullptr;
};

void refresh_panel(PanelState& p) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "Captures this session: %d   |   Burst Mode: %s (%d FPS)",
                g_shot_count.load(), g_burst_active.load() ? "RECORDING" : "Idle", g_cfg.burst_fps);
  SetWindowTextW(p.c_status, widen(buf).c_str());
  std::string last;
  {
    std::lock_guard<std::mutex> lk(g_last_mu);
    last = g_last_file.empty() ? "(none yet — press F12)" : g_last_file;
  }
  SetWindowTextW(p.c_last, widen("Last capture: " + last).c_str());
}

LRESULT CALLBACK panel_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  auto* p = reinterpret_cast<PanelState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
  switch (msg) {
    case WM_CREATE: {
      auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
      p = reinterpret_cast<PanelState*>(cs->lpCreateParams);
      SetWindowLongPtrW(hwnd, GWLP_USERDATA, LONG_PTR(p));
      if (HMODULE dwm = LoadLibraryW(L"dwmapi.dll")) {
        using Fn = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
        if (auto fn = reinterpret_cast<Fn>(GetProcAddress(dwm, "DwmSetWindowAttribute"))) {
          BOOL dark = TRUE;
          if (FAILED(fn(hwnd, 20, &dark, sizeof(dark)))) fn(hwnd, 19, &dark, sizeof(dark));
        }
        FreeLibrary(dwm);
      }
      p->f_ui = CreateFontW(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      p->f_bold = CreateFontW(-14, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      p->f_hdr = CreateFontW(-16, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      p->br_bg = CreateSolidBrush(kBg);

      auto mk = [&](HFONT f, const wchar_t* cls, const wchar_t* txt, DWORD st, int x, int y, int w, int h, int id = 0) {
        HWND c = CreateWindowExW(0, cls, txt, WS_CHILD | WS_VISIBLE | st, x, y, w, h, hwnd, HMENU(INT_PTR(id)),
                                 GetModuleHandleW(nullptr), nullptr);
        SendMessageW(c, WM_SETFONT, WPARAM(f), TRUE);
        return c;
      };

      mk(p->f_hdr, L"STATIC", L"Skate3-VK Clip Studio  (F5)", 0, 18, 14, 380, 24);
      p->c_status = mk(p->f_ui, L"STATIC", L"", 0, 18, 44, 410, 20);
      p->c_last = mk(p->f_ui, L"STATIC", L"", 0, 18, 68, 410, 20);

      mk(p->f_bold, L"BUTTON", L"Screenshot (F12)", BS_OWNERDRAW | WS_TABSTOP, 18, 102, 130, 34, ID_SHOT);
      mk(p->f_bold, L"BUTTON", L"Toggle Burst (F11)", BS_OWNERDRAW | WS_TABSTOP, 156, 102, 135, 34, ID_BURST);
      mk(p->f_bold, L"BUTTON", L"Open Folder", BS_OWNERDRAW | WS_TABSTOP, 299, 102, 125, 34, ID_OPEN_DIR);
      mk(p->f_bold, L"BUTTON", L"Close", BS_OWNERDRAW | WS_TABSTOP, 314, 148, 110, 30, ID_CLOSE);

      refresh_panel(*p);
      SetTimer(hwnd, 1, 400, nullptr);
      return 0;
    }
    case WM_TIMER:
      if (p) refresh_panel(*p);
      return 0;
    case WM_CTLCOLORSTATIC:
      if (!p) break;
      SetBkColor(HDC(wp), kBg);
      SetTextColor(HDC(wp), HWND(lp) == p->c_last ? kMuted : kText);
      return LRESULT(p->br_bg);
    case WM_DRAWITEM: {
      if (!p) break;
      auto* dis = reinterpret_cast<const DRAWITEMSTRUCT*>(lp);
      COLORREF col = kSlate;
      if (dis->CtlID == ID_SHOT) col = kGreen;
      else if (dis->CtlID == ID_BURST) col = kAmber;
      else if (dis->CtlID == ID_OPEN_DIR) col = kAccent;
      HBRUSH br = CreateSolidBrush(col);
      FillRect(dis->hDC, &dis->rcItem, br);
      DeleteObject(br);
      SetBkMode(dis->hDC, TRANSPARENT);
      SetTextColor(dis->hDC, RGB(255, 255, 255));
      SelectObject(dis->hDC, p->f_bold);
      wchar_t txt[64] = {};
      GetWindowTextW(dis->hwndItem, txt, 64);
      RECT r = dis->rcItem;
      DrawTextW(dis->hDC, txt, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      return TRUE;
    }
    case WM_COMMAND:
      if (!p) break;
      if (LOWORD(wp) == ID_SHOT) {
        ShowWindow(hwnd, SW_HIDE);
        Sleep(80);
        take_single_shot();
        ShowWindow(hwnd, SW_SHOW);
        refresh_panel(*p);
      } else if (LOWORD(wp) == ID_BURST) {
        toggle_burst_mode();
        refresh_panel(*p);
      } else if (LOWORD(wp) == ID_OPEN_DIR) {
        std::error_code ec;
        std::filesystem::create_directories(g_cfg.out_dir, ec);
        ShellExecuteW(hwnd, L"open", g_cfg.out_dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
      } else if (LOWORD(wp) == ID_CLOSE) {
        DestroyWindow(hwnd);
      }
      return 0;
    case WM_KEYDOWN:
      if (wp == VK_ESCAPE || wp == VK_F5) { DestroyWindow(hwnd); return 0; }
      break;
    case WM_DESTROY:
      KillTimer(hwnd, 1);
      if (p) {
        DeleteObject(p->f_ui);
        DeleteObject(p->f_bold);
        DeleteObject(p->f_hdr);
        DeleteObject(p->br_bg);
      }
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

void open_panel_async() {
  bool expected = false;
  if (!g_panel_open.compare_exchange_strong(expected, true)) return;
  std::thread([] {
    PanelState st;
    HINSTANCE inst = GetModuleHandleW(nullptr);
    constexpr wchar_t kCls[] = L"Skate3ClipStudioPanel";
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = panel_proc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wc.hbrBackground = CreateSolidBrush(kBg);
    wc.lpszClassName = kCls;
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowExW(WS_EX_TOPMOST, kCls, L"Skate3-VK Clip Studio",
                                WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
                                120, 120, 456, 232, nullptr, nullptr, inst, &st);
    if (hwnd) {
      ShowWindow(hwnd, SW_SHOW);
      UpdateWindow(hwnd);
      MSG msg;
      while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
      }
    }
    g_panel_open.store(false);
  }).detach();
}

void worker_loop() {
  bool p5 = false, p11 = false, p12 = false;
  auto last_burst = std::chrono::steady_clock::now();
  for (;;) {
    const bool c5 = (GetAsyncKeyState(VK_F5) & 0x8000) != 0;
    const bool c11 = (GetAsyncKeyState(VK_F11) & 0x8000) != 0;
    const bool c12 = (GetAsyncKeyState(VK_F12) & 0x8000) != 0;

    if (c5 && !p5) open_panel_async();
    if (c11 && !p11) toggle_burst_mode();
    if (c12 && !p12) take_single_shot();
    p5 = c5; p11 = c11; p12 = c12;

    if (g_burst_active.load()) {
      const auto now = std::chrono::steady_clock::now();
      const int period_ms = 1000 / std::clamp(g_cfg.burst_fps, 1, 15);
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_burst).count() >= period_ms) {
        last_burst = now;
        const auto dest = g_burst_dir / ("frame_" + timestamp_str(true) + ".png");
        capture_png(dest);
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  std::filesystem::path ini_path = "mods/skate3_clip_studio.ini";
  {
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&take_single_shot), &hm)) {
      wchar_t buf[MAX_PATH] = {};
      if (GetModuleFileNameW(hm, buf, MAX_PATH)) {
        ini_path = std::filesystem::path(buf);
        ini_path.replace_extension(".ini");
      }
    }
  }

  cellforge::Config cfg;
  cfg.load(ini_path.string());
  if (!cfg.get_bool("enabled", true)) {
    cellforge::log("skate3_clip_studio: disabled in skate3_clip_studio.ini");
    return 0;
  }

  g_cfg.out_dir = expand_env(cfg.get_string("output_dir", "%USERPROFILE%\\Pictures\\Skate3-VK"));
  g_cfg.burst_fps = std::clamp(cfg.get_int("burst_fps", 5), 1, 15);
  g_cfg.watermark = cfg.get_bool("watermark", true);
  g_cfg.shutter_beep = cfg.get_bool("shutter_beep", true);

  std::thread(worker_loop).detach();
  cellforge::logf("skate3_clip_studio: active (out=%s) — F12 Screenshot, F11 Burst, F5 Panel",
                  g_cfg.out_dir.string().c_str());
  return 0;
}
