// Skate3-VK Online Mod (Path A: Direct P2P / UDP Relay) — v1.0
// Drop `skate3_online.dll` and `skate3_online.ini` into your game's `mods/` folder.
//
// Features:
// - Native Windows Winsock2 UDP networking (no PS3 sys_net or EA Blaze servers required)
// - Works over LAN, Tailscale/ZeroTier, port-forwarding, or the bundled `relay_server.py`
// - 60 Hz skater transform sync + quaternion NLERP & velocity dead-reckoning interpolation
// - Live in-game dark-mode Lobby & Session Panel (press F6 anytime):
//     * Connected players & RTT telemetry
//     * Session chat & Session Marker callout broadcast
//     * Live mode/host/room switching & reconnect without restarting the game
// - Configurable BLUS30464 guest memory bridge for local skater & remote ghost slots

#include "cellforge_mod.h"
#include "net_client.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

namespace {

s3net::NetClient g_net;
cellforge::Hook g_frame_hook;

std::filesystem::path g_ini_path = "mods/skate3_online.ini";
uint32_t g_skater_ptr_addr = 0;
uint32_t g_ghost_pool_addr = 0;
uint32_t g_frame_hook_addr = 0;
uint32_t g_ghost_stride = 0x40;

std::mutex g_feed_mu;
std::deque<std::string> g_feed;

// Headless relay_server.py child process (auto-managed when mode == "host").
std::mutex g_relay_mu;
HANDLE g_relay_proc = nullptr;
HANDLE g_relay_job = nullptr;
uint16_t g_relay_port = 0;
std::atomic<bool> g_relay_active{false};
std::mutex g_ui_cfg_mu;
s3net::ClientConfig g_ui_cfg;

std::atomic<bool> g_ui_open{false};
std::atomic<bool> g_hotkey_running{false};
std::thread g_hotkey_thread;

uint64_t steady_us() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void push_feed(const std::string& line) {
  std::lock_guard<std::mutex> lk(g_feed_mu);
  g_feed.push_back(line);
  while (g_feed.size() > 60) g_feed.pop_front();
}

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
  std::wstring out(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), out.data(), n);
  return out;
}

std::string narrow(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string out(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), out.data(), n, nullptr, nullptr);
  return out;
}

std::string get_ctrl_text(HWND h) {
  int len = GetWindowTextLengthW(h);
  if (len <= 0) return {};
  std::wstring w(size_t(len + 1), L'\0');
  GetWindowTextW(h, w.data(), len + 1);
  w.resize(size_t(len));
  return narrow(w);
}

void save_ini(const s3net::ClientConfig& c, bool show_ui) {
  std::error_code ec;
  if (!g_ini_path.parent_path().empty()) std::filesystem::create_directories(g_ini_path.parent_path(), ec);
  std::ofstream f(g_ini_path);
  if (!f) return;
  f << "# Skate3-VK Online Mod (Path A: Direct P2P / Headless UDP Relay)\n";
  f << "enabled=1\n";
  f << "show_ui_on_start=" << (show_ui ? 1 : 0) << "\n";
  f << "player_name=" << c.player_name << "\n";
  f << "mode=" << c.mode << "\n";
  f << "host=" << c.host << "\n";
  f << "port=" << c.port << "\n";
  f << "room=" << c.room << "\n";
  f << "tick_hz=" << c.tick_hz << "\n";
  char hex[32];
  if (g_skater_ptr_addr) { std::snprintf(hex, sizeof(hex), "%08X", g_skater_ptr_addr); f << "skater_ptr_addr=" << hex << "\n"; }
  else f << "skater_ptr_addr=\n";
  if (g_ghost_pool_addr) { std::snprintf(hex, sizeof(hex), "%08X", g_ghost_pool_addr); f << "ghost_pool_addr=" << hex << "\n"; }
  else f << "ghost_pool_addr=\n";
  if (g_frame_hook_addr) { std::snprintf(hex, sizeof(hex), "%08X", g_frame_hook_addr); f << "frame_hook_addr=" << hex << "\n"; }
  else f << "frame_hook_addr=\n";
}

// ---- Headless relay_server.py management (Option B built into host mode) ----

void stop_headless_relay() {
  std::lock_guard<std::mutex> lk(g_relay_mu);
  if (g_relay_proc) {
    TerminateProcess(g_relay_proc, 0);
    WaitForSingleObject(g_relay_proc, 500);
    CloseHandle(g_relay_proc);
    g_relay_proc = nullptr;
  }
  if (g_relay_job) {
    CloseHandle(g_relay_job);
    g_relay_job = nullptr;
  }
  g_relay_port = 0;
  g_relay_active.store(false);
}

bool start_headless_relay(uint16_t port) {
  {
    std::lock_guard<std::mutex> lk(g_relay_mu);
    if (g_relay_proc && g_relay_port == port && WaitForSingleObject(g_relay_proc, 0) == WAIT_TIMEOUT) {
      g_relay_active.store(true);
      return true;
    }
  }
  stop_headless_relay();

  const std::filesystem::path script = g_ini_path.parent_path() / "relay_server.py";
  std::error_code ec;
  if (!std::filesystem::exists(script, ec)) {
    cellforge::logf("skate3_online: relay_server.py not found at %s (using direct P2P host)", script.string().c_str());
    return false;
  }

  const std::wstring wscript = script.wstring();
  const std::wstring wport = std::to_wstring(port);
  const std::wstring wdir = script.parent_path().wstring();
  const wchar_t* launchers[] = {
      L"pythonw.exe",
      L"python.exe",
      L"py.exe -3",
  };

  for (const wchar_t* exe : launchers) {
    std::wstring cmd = std::wstring(exe) + L" \"" + wscript + L"\" --port " + wport;
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB, nullptr,
                        wdir.empty() ? nullptr : wdir.c_str(), &si, &pi)) {
      // Retry without CREATE_BREAKAWAY_FROM_JOB in case the parent job disallows breakaway.
      if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE,
                          CREATE_NO_WINDOW, nullptr,
                          wdir.empty() ? nullptr : wdir.c_str(), &si, &pi)) {
        continue;
      }
    }

    // Bind child to a Job Object with KILL_ON_JOB_CLOSE so closing the game never leaves an orphan relay.
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
      jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      SetInformationJobObject(job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
      AssignProcessToJobObject(job, pi.hProcess);
    }
    CloseHandle(pi.hThread);

    // Wait briefly to ensure relay_server.py stays up and binds the UDP port.
    if (WaitForSingleObject(pi.hProcess, 180) == WAIT_TIMEOUT) {
      std::lock_guard<std::mutex> lk(g_relay_mu);
      g_relay_proc = pi.hProcess;
      g_relay_job = job;
      g_relay_port = port;
      g_relay_active.store(true);
      cellforge::logf("skate3_online: started headless relay_server.py on UDP :%u", unsigned(port));
      push_feed("[relay] Started headless relay on UDP :" + std::to_string(port));
      return true;
    }

    // Process exited immediately (e.g. launcher stub or port in use); clean up and try next.
    CloseHandle(pi.hProcess);
    if (job) CloseHandle(job);
  }

  cellforge::log("skate3_online: could not launch headless relay_server.py (Python not found); falling back to direct P2P host");
  return false;
}

bool start_session(const s3net::ClientConfig& ui_cfg) {
  {
    std::lock_guard<std::mutex> lk(g_ui_cfg_mu);
    g_ui_cfg = ui_cfg;
  }

  s3net::ClientConfig eff = ui_cfg;
  if (ui_cfg.mode == "host") {
    // Stop client first so UDP port is free for the headless relay server to bind.
    g_net.stop();
    if (start_headless_relay(ui_cfg.port)) {
      eff.mode = "relay";
      eff.host = "127.0.0.1";
    }
  } else {
    stop_headless_relay();
  }

  return g_net.start(eff, [](const std::string& m) {
    cellforge::logf("skate3_online: %s", m.c_str());
    push_feed("[net] " + m);
  });
}

s3net::ClientConfig get_ui_cfg() {
  std::lock_guard<std::mutex> lk(g_ui_cfg_mu);
  return g_ui_cfg;
}

// ---- Guest RAM Sync ---------------------------------------------------------

void sync_skater_tick() {
  if (g_skater_ptr_addr != 0) {
    const uint32_t entity = cellforge::read_u32(g_skater_ptr_addr);
    if (entity >= 0x00100000u && entity < 0x50000000u) {
      s3net::SkaterState st{};
      for (int i = 0; i < 3; ++i) st.pos[i] = cellforge::read_f32(entity + i * 4);
      for (int i = 0; i < 3; ++i) st.vel[i] = cellforge::read_f32(entity + 12 + i * 4);
      for (int i = 0; i < 4; ++i) st.quat[i] = cellforge::read_f32(entity + 24 + i * 4);
      g_net.set_local_state(st);
    }
  }

  if (g_ghost_pool_addr != 0) {
    const uint64_t now = steady_us();
    const auto peers = g_net.get_remote_peers();
    uint32_t slot = 0;
    for (const auto& p : peers) {
      if (slot >= s3net::kMaxPlayers - 1) break;
      const s3net::SkaterState interp = s3net::NetClient::interpolate(p, now);
      const uint32_t dst = g_ghost_pool_addr + slot * g_ghost_stride;
      for (int i = 0; i < 3; ++i) cellforge::write_f32(dst + i * 4, interp.pos[i]);
      for (int i = 0; i < 3; ++i) cellforge::write_f32(dst + 12 + i * 4, interp.vel[i]);
      for (int i = 0; i < 4; ++i) cellforge::write_f32(dst + 24 + i * 4, interp.quat[i]);
      ++slot;
    }
  }
}

void hooked_frame_update(ppu::Context& c) {
  g_frame_hook.call_original(c);
  sync_skater_tick();
}

// ---- F6 Dark-Mode Lobby & Session Panel ------------------------------------

constexpr COLORREF kBg = RGB(18, 21, 28);
constexpr COLORREF kCard = RGB(27, 32, 43);
constexpr COLORREF kText = RGB(229, 233, 240);
constexpr COLORREF kMuted = RGB(148, 163, 184);
constexpr COLORREF kAccent = RGB(37, 99, 235);
constexpr COLORREF kGreen = RGB(16, 185, 129);
constexpr COLORREF kAmber = RGB(217, 119, 6);
constexpr COLORREF kSlate = RGB(51, 65, 85);

enum : int {
  ID_MODE = 301,
  ID_HOST,
  ID_PORT,
  ID_ROOM,
  ID_NAME,
  ID_RECONNECT,
  ID_PEERS,
  ID_FEED,
  ID_CHAT_IN,
  ID_SEND_CHAT,
  ID_DROP_MARKER,
  ID_CLOSE_UI,
  TIMER_REFRESH = 1,
};

struct LobbyUi {
  HWND hwnd = nullptr;
  HWND c_status = nullptr, c_mode = nullptr, c_host = nullptr, c_port = nullptr, c_room = nullptr, c_name = nullptr;
  HWND c_peers = nullptr, c_feed = nullptr, c_chat = nullptr;
  HFONT f_ui = nullptr, f_bold = nullptr, f_hdr = nullptr, f_mono = nullptr;
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

HWND mk(HWND parent, HFONT font, const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y, int w, int h, int id = 0) {
  HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, parent, HMENU(INT_PTR(id)),
                           GetModuleHandleW(nullptr), nullptr);
  SendMessageW(c, WM_SETFONT, WPARAM(font), TRUE);
  return c;
}

void refresh_lobby_ui(LobbyUi& u) {
  const auto cfg = g_net.config();
  const auto ucfg = get_ui_cfg();
  const auto peers = g_net.get_remote_peers();
  char buf[256];
  if (!g_net.running()) {
    std::snprintf(buf, sizeof(buf), "Status: Disconnected");
  } else if (g_relay_active.load()) {
    std::snprintf(buf, sizeof(buf), "Hosting (Headless Relay) UDP :%u | Room %s | Players: %zu | RTT %u ms",
                  unsigned(ucfg.port), ucfg.room.c_str(), peers.size() + 1, unsigned(g_net.rtt_ms()));
  } else if (cfg.mode == "host") {
    std::snprintf(buf, sizeof(buf), "Hosting UDP :%u | Room %s | Players: %zu | TX %llu / RX %llu",
                  unsigned(cfg.port), cfg.room.c_str(), peers.size() + 1,
                  static_cast<unsigned long long>(g_net.packets_sent()),
                  static_cast<unsigned long long>(g_net.packets_recv()));
  } else if (g_net.local_peer_id() != 0) {
    std::snprintf(buf, sizeof(buf), "Connected (%s) | Peer #%u | Room %s | RTT %u ms | Players: %zu",
                  ucfg.mode.c_str(), unsigned(g_net.local_peer_id()), cfg.room.c_str(),
                  unsigned(g_net.rtt_ms()), peers.size() + 1);
  } else {
    std::snprintf(buf, sizeof(buf), "Connecting (%s -> %s:%u, Room %s)...",
                  ucfg.mode.c_str(), cfg.host.c_str(), unsigned(cfg.port), cfg.room.c_str());
  }
  SetWindowTextW(u.c_status, widen(buf).c_str());

  SendMessageW(u.c_peers, LB_RESETCONTENT, 0, 0);
  char self_line[128];
  std::snprintf(self_line, sizeof(self_line), "  [YOU #%u]  %s", unsigned(g_net.local_peer_id()), cfg.player_name.c_str());
  SendMessageW(u.c_peers, LB_ADDSTRING, 0, LPARAM(widen(self_line).c_str()));
  for (const auto& p : peers) {
    char pline[192];
    const std::string nm = p.name.empty() ? ("Peer #" + std::to_string(p.peer_id)) : p.name;
    std::snprintf(pline, sizeof(pline), "  [#%u]  %-18s  pos=(%.1f, %.1f, %.1f)",
                  unsigned(p.peer_id), nm.c_str(), p.curr_state.pos[0], p.curr_state.pos[1], p.curr_state.pos[2]);
    SendMessageW(u.c_peers, LB_ADDSTRING, 0, LPARAM(widen(pline).c_str()));
  }

  std::string feed_txt;
  {
    std::lock_guard<std::mutex> lk(g_feed_mu);
    for (const auto& l : g_feed) {
      feed_txt += l;
      feed_txt += "\r\n";
    }
  }
  const int old_len = GetWindowTextLengthW(u.c_feed);
  const std::wstring wfeed = widen(feed_txt);
  if (int(wfeed.size()) != old_len) {
    SetWindowTextW(u.c_feed, wfeed.c_str());
    SendMessageW(u.c_feed, EM_SETSEL, wfeed.size(), wfeed.size());
    SendMessageW(u.c_feed, EM_SCROLLCARET, 0, 0);
  }
}

LRESULT CALLBACK lobby_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  auto* u = reinterpret_cast<LobbyUi*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
  switch (msg) {
    case WM_CREATE: {
      auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
      u = reinterpret_cast<LobbyUi*>(cs->lpCreateParams);
      u->hwnd = hwnd;
      SetWindowLongPtrW(hwnd, GWLP_USERDATA, LONG_PTR(u));
      enable_dark_bar(hwnd);

      u->f_ui = CreateFontW(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      u->f_bold = CreateFontW(-14, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      u->f_hdr = CreateFontW(-16, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, VARIABLE_PITCH, L"Segoe UI");
      u->f_mono = CreateFontW(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, FIXED_PITCH, L"Consolas");
      u->br_bg = CreateSolidBrush(kBg);
      u->br_card = CreateSolidBrush(kCard);

      const auto cfg = get_ui_cfg();
      const int lx = 18;
      mk(hwnd, u->f_hdr, L"STATIC", L"Skate3-VK Online  (F6 to toggle)", 0, lx, 12, 480, 24);
      u->c_status = mk(hwnd, u->f_ui, L"STATIC", L"Status: Initializing...", 0, lx, 38, 500, 20);

      // Connection bar
      mk(hwnd, u->f_ui, L"STATIC", L"Mode:", 0, lx, 68, 46, 20);
      u->c_mode = mk(hwnd, u->f_ui, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, lx + 48, 64, 90, 120, ID_MODE);
      SendMessageW(u->c_mode, CB_ADDSTRING, 0, LPARAM(L"relay"));
      SendMessageW(u->c_mode, CB_ADDSTRING, 0, LPARAM(L"host"));
      SendMessageW(u->c_mode, CB_ADDSTRING, 0, LPARAM(L"join"));
      int mi = (cfg.mode == "host") ? 1 : (cfg.mode == "join" ? 2 : 0);
      SendMessageW(u->c_mode, CB_SETCURSEL, mi, 0);

      mk(hwnd, u->f_ui, L"STATIC", L"Host:", 0, lx + 148, 68, 38, 20);
      u->c_host = mk(hwnd, u->f_ui, L"EDIT", widen(cfg.host).c_str(), WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, lx + 188, 64, 130, 24, ID_HOST);
      mk(hwnd, u->f_ui, L"STATIC", L"Port:", 0, lx + 328, 68, 36, 20);
      u->c_port = mk(hwnd, u->f_ui, L"EDIT", std::to_wstring(cfg.port).c_str(), WS_BORDER | ES_NUMBER | WS_TABSTOP, lx + 366, 64, 64, 24, ID_PORT);

      mk(hwnd, u->f_ui, L"STATIC", L"Room:", 0, lx, 100, 46, 20);
      u->c_room = mk(hwnd, u->f_ui, L"EDIT", widen(cfg.room).c_str(), WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, lx + 48, 96, 90, 24, ID_ROOM);
      mk(hwnd, u->f_ui, L"STATIC", L"Name:", 0, lx + 148, 100, 44, 20);
      u->c_name = mk(hwnd, u->f_ui, L"EDIT", widen(cfg.player_name).c_str(), WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, lx + 194, 96, 124, 24, ID_NAME);
      mk(hwnd, u->f_bold, L"BUTTON", L"Reconnect", BS_OWNERDRAW | WS_TABSTOP, lx + 328, 95, 102, 26, ID_RECONNECT);

      // Connected players
      mk(hwnd, u->f_bold, L"STATIC", L"Players in Session:", 0, lx, 132, 200, 20);
      u->c_peers = mk(hwnd, u->f_mono, L"LISTBOX", L"", WS_BORDER | WS_VSCROLL | LBS_NOINTEGRALHEIGHT, lx, 154, 495, 105, ID_PEERS);

      // Session feed / chat
      mk(hwnd, u->f_bold, L"STATIC", L"Session Feed && Chat:", 0, lx, 268, 220, 20);
      u->c_feed = mk(hwnd, u->f_mono, L"EDIT", L"", WS_BORDER | WS_VSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL, lx, 290, 495, 145, ID_FEED);

      u->c_chat = mk(hwnd, u->f_ui, L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, lx, 444, 290, 28, ID_CHAT_IN);
      mk(hwnd, u->f_bold, L"BUTTON", L"Send Chat", BS_OWNERDRAW | WS_TABSTOP, lx + 298, 444, 95, 28, ID_SEND_CHAT);
      mk(hwnd, u->f_bold, L"BUTTON", L"Drop Marker", BS_OWNERDRAW | WS_TABSTOP, lx + 400, 444, 113, 28, ID_DROP_MARKER);

      mk(hwnd, u->f_bold, L"BUTTON", L"Close (F6)", BS_OWNERDRAW | WS_TABSTOP, lx + 390, 482, 123, 30, ID_CLOSE_UI);

      refresh_lobby_ui(*u);
      SetTimer(hwnd, TIMER_REFRESH, 500, nullptr);
      return 0;
    }
    case WM_TIMER:
      if (u && wp == TIMER_REFRESH) refresh_lobby_ui(*u);
      return 0;
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLORBTN: {
      if (!u) break;
      HDC hdc = HDC(wp);
      SetBkColor(hdc, kBg);
      SetTextColor(hdc, HWND(lp) == u->c_status ? kMuted : kText);
      return LRESULT(u->br_bg);
    }
    case WM_CTLCOLORLISTBOX:
    case WM_CTLCOLOREDIT: {
      if (!u) break;
      HDC hdc = HDC(wp);
      SetBkColor(hdc, kCard);
      SetTextColor(hdc, kText);
      return LRESULT(u->br_card);
    }
    case WM_DRAWITEM: {
      if (!u) break;
      auto* dis = reinterpret_cast<const DRAWITEMSTRUCT*>(lp);
      COLORREF col = kSlate;
      if (dis->CtlID == ID_RECONNECT) col = kGreen;
      else if (dis->CtlID == ID_SEND_CHAT) col = kAccent;
      else if (dis->CtlID == ID_DROP_MARKER) col = kAmber;
      HBRUSH br = CreateSolidBrush(col);
      FillRect(dis->hDC, &dis->rcItem, br);
      DeleteObject(br);
      SetBkMode(dis->hDC, TRANSPARENT);
      SetTextColor(dis->hDC, RGB(255, 255, 255));
      SelectObject(dis->hDC, u->f_bold);
      wchar_t txt[64] = {};
      GetWindowTextW(dis->hwndItem, txt, 64);
      RECT r = dis->rcItem;
      DrawTextW(dis->hDC, txt, -1, &r, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      return TRUE;
    }
    case WM_COMMAND: {
      if (!u) break;
      const int id = LOWORD(wp);
      if (id == ID_RECONNECT) {
        s3net::ClientConfig nc = get_ui_cfg();
        const int mi = int(SendMessageW(u->c_mode, CB_GETCURSEL, 0, 0));
        nc.mode = (mi == 1) ? "host" : (mi == 2 ? "join" : "relay");
        nc.host = get_ctrl_text(u->c_host);
        if (nc.host.empty()) nc.host = "127.0.0.1";
        const int p = std::atoi(get_ctrl_text(u->c_port).c_str());
        nc.port = static_cast<uint16_t>((p > 0 && p <= 65535) ? p : s3net::kDefaultPort);
        nc.room = get_ctrl_text(u->c_room);
        if (nc.room.empty()) nc.room = "SK8A";
        nc.player_name = get_ctrl_text(u->c_name);
        if (nc.player_name.empty()) nc.player_name = "Skater";

        save_ini(nc, false);
        push_feed("[sys] Reconnecting (" + nc.mode + " " + nc.host + ":" + std::to_string(nc.port) + " room=" + nc.room + ")...");
        start_session(nc);
        refresh_lobby_ui(*u);
      } else if (id == ID_SEND_CHAT) {
        std::string txt = get_ctrl_text(u->c_chat);
        if (!txt.empty()) {
          const std::string full = g_net.config().player_name + ": " + txt;
          g_net.send_event(static_cast<uint16_t>(s3net::EventKind::Chat), full);
          push_feed("[chat] " + full);
          SetWindowTextW(u->c_chat, L"");
          refresh_lobby_ui(*u);
        }
      } else if (id == ID_DROP_MARKER) {
        const std::string msg = g_net.config().player_name + " dropped a Session Marker!";
        g_net.send_event(static_cast<uint16_t>(s3net::EventKind::MarkerTeleport), msg);
        push_feed("[marker] " + msg);
        refresh_lobby_ui(*u);
      } else if (id == ID_CLOSE_UI) {
        DestroyWindow(hwnd);
      }
      return 0;
    }
    case WM_KEYDOWN:
      if (wp == VK_ESCAPE || wp == VK_F6) { DestroyWindow(hwnd); return 0; }
      break;
    case WM_DESTROY:
      KillTimer(hwnd, TIMER_REFRESH);
      if (u) {
        DeleteObject(u->f_ui);
        DeleteObject(u->f_bold);
        DeleteObject(u->f_hdr);
        DeleteObject(u->f_mono);
        DeleteObject(u->br_bg);
        DeleteObject(u->br_card);
      }
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

void open_lobby_window_async() {
  bool expected = false;
  if (!g_ui_open.compare_exchange_strong(expected, true)) return;
  std::thread([] {
    LobbyUi ui;
    HINSTANCE inst = GetModuleHandleW(nullptr);
    constexpr wchar_t kCls[] = L"Skate3OnlineLobbyWnd";
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = lobby_proc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wc.hbrBackground = CreateSolidBrush(kBg);
    wc.lpszClassName = kCls;
    RegisterClassExW(&wc);

    const int W = 548, H = 560;
    HWND hwnd = CreateWindowExW(WS_EX_TOPMOST, kCls, L"Skate3-VK Online — Session Lobby",
                                WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
                                80, 80, W, H, nullptr, nullptr, inst, &ui);
    if (hwnd) {
      ShowWindow(hwnd, SW_SHOW);
      UpdateWindow(hwnd);
      MSG msg;
      while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(hwnd, &msg)) {
          TranslateMessage(&msg);
          DispatchMessageW(&msg);
        }
      }
    }
    g_ui_open.store(false);
  }).detach();
}

void start_hotkey_listener() {
  if (g_hotkey_running.exchange(true)) return;
  g_hotkey_thread = std::thread([] {
    bool prev_f6 = false;
    while (g_hotkey_running.load()) {
      const bool cur_f6 = (GetAsyncKeyState(VK_F6) & 0x8000) != 0;
      if (cur_f6 && !prev_f6) open_lobby_window_async();
      prev_f6 = cur_f6;
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
  });
  g_hotkey_thread.detach();
}

uint32_t parse_hex(const std::string& s) {
  if (s.empty()) return 0;
  return static_cast<uint32_t>(std::strtoul(s.c_str(), nullptr, 16));
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  // Locate mods/skate3_online.ini relative to this DLL if possible.
  {
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&sync_skater_tick), &hm)) {
      wchar_t buf[MAX_PATH] = {};
      if (GetModuleFileNameW(hm, buf, MAX_PATH)) {
        g_ini_path = std::filesystem::path(buf);
        g_ini_path.replace_extension(".ini");
      }
    }
  }

  cellforge::Config cfg;
  const bool loaded = cfg.load(g_ini_path.string());
  if (!cfg.get_bool("enabled", true)) {
    cellforge::log("skate3_online: disabled in skate3_online.ini");
    return 0;
  }

  s3net::ClientConfig net_cfg;
  net_cfg.mode = cfg.get_string("mode", "relay");
  net_cfg.host = cfg.get_string("host", "127.0.0.1");
  net_cfg.port = static_cast<uint16_t>(cfg.get_int("port", s3net::kDefaultPort));
  net_cfg.room = cfg.get_string("room", "SK8A");
  net_cfg.player_name = cfg.get_string("player_name", "Skater");
  net_cfg.tick_hz = static_cast<uint16_t>(cfg.get_int("tick_hz", 60));
  const bool show_ui = cfg.get_bool("show_ui_on_start", false);

  g_skater_ptr_addr = parse_hex(cfg.get_string("skater_ptr_addr", ""));
  g_ghost_pool_addr = parse_hex(cfg.get_string("ghost_pool_addr", ""));
  g_frame_hook_addr = parse_hex(cfg.get_string("frame_hook_addr", ""));

  cellforge::logf("skate3_online: starting (config=%s, mode=%s, host=%s:%u, room=%s) — press F6 for Lobby UI",
                  loaded ? g_ini_path.string().c_str() : "defaults",
                  net_cfg.mode.c_str(), net_cfg.host.c_str(), unsigned(net_cfg.port), net_cfg.room.c_str());

  g_net.set_event_callback([](uint8_t peer_id, uint16_t kind, const std::string& payload) {
    if (kind == static_cast<uint16_t>(s3net::EventKind::Chat)) {
      push_feed("[chat] " + payload);
    } else if (kind == static_cast<uint16_t>(s3net::EventKind::MarkerTeleport)) {
      push_feed("[marker] " + payload);
    } else if (kind == static_cast<uint16_t>(s3net::EventKind::TrickCallout)) {
      push_feed("[trick #" + std::to_string(peer_id) + "] " + payload);
    }
  });

  if (!start_session(net_cfg)) {
    cellforge::log("skate3_online: failed to start UDP transport");
    return 1;
  }

  if (g_frame_hook_addr != 0) {
    if (g_frame_hook.install(g_frame_hook_addr, hooked_frame_update)) {
      cellforge::logf("skate3_online: hooked frame tick at 0x%08X", g_frame_hook_addr);
    } else {
      cellforge::logf("skate3_online: warning: frame_hook_addr 0x%08X is not a recompiled function", g_frame_hook_addr);
    }
  }

  start_hotkey_listener();
  if (show_ui) open_lobby_window_async();
  return 0;
}
