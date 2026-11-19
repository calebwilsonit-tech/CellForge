// Skate3-VK Speedrun Split Timer & Line Tracker HUD (CellForge Mod)
// Hotkeys: F7 = Show/Hide HUD, F8 = Start/Split, F9 = Pause/Resume, F10 = Reset.

#include "cellforge_mod.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct SplitRow {
  std::string name;
  double cur_sec = -1.0;
  double pb_sec = -1.0;
};

struct TimerState {
  std::mutex mu;
  std::vector<SplitRow> splits;
  int current_split = 0;
  bool running = false;
  bool paused = false;
  bool finished = false;
  double accumulated_sec = 0.0;
  Clock::time_point start_tp{};
  double best_total_sec = -1.0;
  std::filesystem::path pb_path;
  int opacity = 225;
  bool visible = true;
};

TimerState g_st;
std::atomic<HWND> g_hud_hwnd{nullptr};

double elapsed_locked() {
  if (!g_st.running) return g_st.accumulated_sec;
  if (g_st.paused) return g_st.accumulated_sec;
  const double dt = std::chrono::duration<double>(Clock::now() - g_st.start_tp).count();
  return g_st.accumulated_sec + dt;
}

std::string fmt_time(double sec) {
  if (sec < 0.0) return "--:--.---";
  const int total_ms = int(sec * 1000.0);
  const int mins = total_ms / 60000;
  const int secs = (total_ms / 1000) % 60;
  const int ms = total_ms % 1000;
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%02d:%02d.%03d", mins, secs, ms);
  return buf;
}

std::string fmt_delta(double delta) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%+.2fs", delta);
  return buf;
}

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
  std::wstring out(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), out.data(), n);
  return out;
}

void load_pb_locked() {
  std::ifstream f(g_st.pb_path);
  if (!f) return;
  std::string line;
  if (std::getline(f, line)) g_st.best_total_sec = std::atof(line.c_str());
  for (size_t i = 0; i < g_st.splits.size() && std::getline(f, line); ++i) {
    g_st.splits[i].pb_sec = std::atof(line.c_str());
  }
}

void save_pb_locked() {
  std::ofstream f(g_st.pb_path);
  if (!f) return;
  f << g_st.best_total_sec << "\n";
  for (const auto& s : g_st.splits) f << s.pb_sec << "\n";
}

void action_start_or_split() {
  std::lock_guard<std::mutex> lk(g_st.mu);
  if (g_st.finished) return;
  if (!g_st.running) {
    g_st.running = true;
    g_st.paused = false;
    g_st.accumulated_sec = 0.0;
    g_st.current_split = 0;
    g_st.start_tp = Clock::now();
    for (auto& s : g_st.splits) s.cur_sec = -1.0;
    return;
  }
  if (g_st.paused) return;
  const double now_sec = elapsed_locked();
  if (g_st.current_split < int(g_st.splits.size())) {
    g_st.splits[size_t(g_st.current_split)].cur_sec = now_sec;
    g_st.current_split++;
  }
  if (g_st.current_split >= int(g_st.splits.size())) {
    g_st.accumulated_sec = now_sec;
    g_st.running = false;
    g_st.finished = true;
    if (g_st.best_total_sec < 0.0 || now_sec < g_st.best_total_sec) {
      g_st.best_total_sec = now_sec;
      for (auto& s : g_st.splits) s.pb_sec = s.cur_sec;
      save_pb_locked();
      cellforge::logf("skate3_trick_timer: NEW PB! %s", fmt_time(now_sec).c_str());
    }
  }
}

void action_pause_resume() {
  std::lock_guard<std::mutex> lk(g_st.mu);
  if (!g_st.running) return;
  if (!g_st.paused) {
    g_st.accumulated_sec = elapsed_locked();
    g_st.paused = true;
  } else {
    g_st.start_tp = Clock::now();
    g_st.paused = false;
  }
}

void action_reset() {
  std::lock_guard<std::mutex> lk(g_st.mu);
  g_st.running = false;
  g_st.paused = false;
  g_st.finished = false;
  g_st.accumulated_sec = 0.0;
  g_st.current_split = 0;
  for (auto& s : g_st.splits) s.cur_sec = -1.0;
}

constexpr COLORREF kBg = RGB(15, 18, 25);
constexpr COLORREF kHeader = RGB(24, 30, 42);
constexpr COLORREF kBorder = RGB(51, 65, 85);
constexpr COLORREF kText = RGB(232, 236, 244);
constexpr COLORREF kMuted = RGB(148, 163, 184);
constexpr COLORREF kGreen = RGB(34, 197, 94);
constexpr COLORREF kRed = RGB(239, 68, 68);
constexpr COLORREF kGold = RGB(250, 204, 21);
constexpr COLORREF kAccent = RGB(59, 130, 246);

LRESULT CALLBACK hud_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  static HFONT f_title = nullptr, f_big = nullptr, f_row = nullptr, f_tiny = nullptr;
  switch (msg) {
    case WM_CREATE:
      f_title = CreateFontW(-13, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      f_big = CreateFontW(-30, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
      f_row = CreateFontW(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
      f_tiny = CreateFontW(-11, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      SetTimer(hwnd, 1, 33, nullptr);
      return 0;
    case WM_NCHITTEST:
      return HTCAPTION;  // Click anywhere to drag the HUD
    case WM_TIMER:
      InvalidateRect(hwnd, nullptr, FALSE);
      return 0;
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hwnd, &ps);
      RECT rc;
      GetClientRect(hwnd, &rc);

      HDC mem = CreateCompatibleDC(hdc);
      HBITMAP bmp = CreateCompatibleBitmap(hdc, rc.right, rc.bottom);
      HGDIOBJ old_bmp = SelectObject(mem, bmp);

      HBRUSH br_bg = CreateSolidBrush(kBg);
      FillRect(mem, &rc, br_bg);
      DeleteObject(br_bg);

      RECT hdr{0, 0, rc.right, 28};
      HBRUSH br_hdr = CreateSolidBrush(kHeader);
      FillRect(mem, &hdr, br_hdr);
      DeleteObject(br_hdr);

      HPEN pen = CreatePen(PS_SOLID, 1, kBorder);
      HGDIOBJ old_pen = SelectObject(mem, pen);
      HGDIOBJ old_br = SelectObject(mem, GetStockObject(NULL_BRUSH));
      Rectangle(mem, 0, 0, rc.right, rc.bottom);
      SelectObject(mem, old_pen);
      SelectObject(mem, old_br);
      DeleteObject(pen);

      SetBkMode(mem, TRANSPARENT);
      SelectObject(mem, f_title);
      SetTextColor(mem, kAccent);
      RECT tr{10, 5, rc.right - 10, 24};
      DrawTextW(mem, L"SKATE3-VK  SPLIT TIMER", -1, &tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

      double now_sec = 0.0;
      double best_sec = -1.0;
      bool running = false, paused = false, finished = false;
      int cur_idx = 0;
      std::vector<SplitRow> rows;
      {
        std::lock_guard<std::mutex> lk(g_st.mu);
        now_sec = elapsed_locked();
        best_sec = g_st.best_total_sec;
        running = g_st.running;
        paused = g_st.paused;
        finished = g_st.finished;
        cur_idx = g_st.current_split;
        rows = g_st.splits;
      }

      // Main big clock
      SelectObject(mem, f_big);
      SetTextColor(mem, finished ? kGold : (paused ? kMuted : (running ? kGreen : kText)));
      RECT clk{10, 34, rc.right - 10, 72};
      DrawTextW(mem, widen(fmt_time(now_sec)).c_str(), -1, &clk, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);

      // Splits list
      SelectObject(mem, f_row);
      int y = 78;
      for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (int(i) == cur_idx && running) {
          RECT hi{4, y - 2, rc.right - 4, y + 18};
          HBRUSH b = CreateSolidBrush(RGB(30, 41, 59));
          FillRect(mem, &hi, b);
          DeleteObject(b);
        }
        SetTextColor(mem, int(i) == cur_idx ? kText : kMuted);
        RECT nr{10, y, rc.right - 135, y + 18};
        DrawTextW(mem, widen(r.name).c_str(), -1, &nr, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);

        if (r.cur_sec >= 0.0 && r.pb_sec >= 0.0) {
          const double d = r.cur_sec - r.pb_sec;
          SetTextColor(mem, d <= 0.0 ? kGreen : kRed);
          RECT dr{rc.right - 132, y, rc.right - 74, y + 18};
          DrawTextW(mem, widen(fmt_delta(d)).c_str(), -1, &dr, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        }

        SetTextColor(mem, r.cur_sec >= 0.0 ? kText : kMuted);
        const double shown = (r.cur_sec >= 0.0) ? r.cur_sec : r.pb_sec;
        RECT sr{rc.right - 72, y, rc.right - 10, y + 18};
        DrawTextW(mem, widen(fmt_time(shown)).c_str(), -1, &sr, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
        y += 20;
      }

      // Footer
      SelectObject(mem, f_tiny);
      SetTextColor(mem, kMuted);
      std::string foot = "F7 Hide | F8 Split | F9 Pause | F10 Reset";
      if (best_sec >= 0.0) foot = "PB: " + fmt_time(best_sec) + "  |  F8 Split  F10 Reset";
      RECT fr{10, rc.bottom - 20, rc.right - 10, rc.bottom - 4};
      DrawTextW(mem, widen(foot).c_str(), -1, &fr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);

      BitBlt(hdc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
      SelectObject(mem, old_bmp);
      DeleteObject(bmp);
      DeleteDC(mem);
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_DESTROY:
      KillTimer(hwnd, 1);
      DeleteObject(f_title);
      DeleteObject(f_big);
      DeleteObject(f_row);
      DeleteObject(f_tiny);
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

void hud_thread_main() {
  HINSTANCE inst = GetModuleHandleW(nullptr);
  constexpr wchar_t kCls[] = L"Skate3TrickTimerHud";
  WNDCLASSEXW wc{sizeof(wc)};
  wc.lpfnWndProc = hud_proc;
  wc.hInstance = inst;
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  wc.lpszClassName = kCls;
  RegisterClassExW(&wc);

  int h = 112 + int(g_st.splits.size()) * 20;
  HWND hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW, kCls, L"Skate3-VK Split Timer",
                              WS_POPUP, 24, 48, 300, h, nullptr, nullptr, inst, nullptr);
  if (!hwnd) return;
  SetLayeredWindowAttributes(hwnd, 0, BYTE(std::clamp(g_st.opacity, 80, 255)), LWA_ALPHA);
  g_hud_hwnd.store(hwnd);
  if (g_st.visible) ShowWindow(hwnd, SW_SHOWNOACTIVATE);

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
}

void hotkey_thread_main() {
  bool p7 = false, p8 = false, p9 = false, p10 = false;
  for (;;) {
    const bool c7 = (GetAsyncKeyState(VK_F7) & 0x8000) != 0;
    const bool c8 = (GetAsyncKeyState(VK_F8) & 0x8000) != 0;
    const bool c9 = (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
    const bool c10 = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
    if (c7 && !p7) {
      HWND h = g_hud_hwnd.load();
      if (h) {
        const bool vis = IsWindowVisible(h) != FALSE;
        ShowWindow(h, vis ? SW_HIDE : SW_SHOWNOACTIVATE);
      }
    }
    if (c8 && !p8) action_start_or_split();
    if (c9 && !p9) action_pause_resume();
    if (c10 && !p10) action_reset();
    p7 = c7; p8 = c8; p9 = c9; p10 = c10;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  std::filesystem::path ini_path = "mods/skate3_trick_timer.ini";
  {
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&action_reset), &hm)) {
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
    cellforge::log("skate3_trick_timer: disabled in skate3_trick_timer.ini");
    return 0;
  }

  g_st.visible = cfg.get_bool("show_on_start", true);
  g_st.opacity = cfg.get_int("opacity", 225);
  g_st.pb_path = ini_path.parent_path() / "skate3_trick_timer_pb.txt";

  const std::string splits_csv = cfg.get_string(
      "splits", "Super Ultra Mega Park,Carverton Quarry,Observatory,University Ditch,Downtown Plaza");
  std::stringstream ss(splits_csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.erase(item.begin());
    while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) item.pop_back();
    if (!item.empty()) g_st.splits.push_back(SplitRow{item});
  }
  if (g_st.splits.empty()) g_st.splits.push_back(SplitRow{"Run Finish"});

  {
    std::lock_guard<std::mutex> lk(g_st.mu);
    load_pb_locked();
  }

  std::thread(hud_thread_main).detach();
  std::thread(hotkey_thread_main).detach();
  cellforge::logf("skate3_trick_timer: active (%zu splits) — F7 Hide/Show, F8 Split, F9 Pause, F10 Reset",
                  g_st.splits.size());
  return 0;
}
