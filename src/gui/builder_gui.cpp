// CellForge Installer & Launcher (CellForge-Setup.exe).
// Single self-contained executable:
// - Checks & automatically downloads/installs prerequisites (CellForge source from GitHub,
//   Visual Studio 2022 C++ Build Tools + CMake, and ps3recomp.exe).
// - Lets the user choose their Game Folder, Decrypted ELF, and standalone Install Folder.
// - Builds and deploys a completely detached game (e.g. Skate3-VK.exe + .ini + mods/ +
//   embedded ICON0.PNG icon) with persistent saves in %USERPROFILE%\Saved Games.
// This program does not decrypt anything; an encrypted EBOOT.BIN is rejected.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dwmapi.h>
#include <objidl.h>
#include <gdiplus.h>
#include <shellapi.h>
#include <shobjidl.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "builder_core.h"

namespace {

using builder::fs::path;

enum : int {
  ID_GAME = 101,
  ID_GAME_BROWSE,
  ID_ELF,
  ID_ELF_BROWSE,
  ID_INSTALL,
  ID_INSTALL_BROWSE,
  ID_BUILD,
  ID_UPDATE,
  ID_PREREQS,
  ID_CANCEL,
  ID_PLAY,
  ID_SAVES,
  ID_LOG,
  ID_STATUS,
  ID_PREREQ_STATUS,
};
constexpr UINT WM_APP_LOG = WM_APP + 1;   // lParam: std::string*
constexpr UINT WM_APP_DONE = WM_APP + 2;  // wParam: success

// Dark slate palette
constexpr COLORREF kColBg = RGB(18, 21, 28);
constexpr COLORREF kColCard = RGB(27, 32, 43);
constexpr COLORREF kColText = RGB(229, 233, 240);
constexpr COLORREF kColMuted = RGB(148, 163, 184);
constexpr COLORREF kColBlue = RGB(37, 99, 235);
constexpr COLORREF kColSky = RGB(14, 165, 233);
constexpr COLORREF kColGreen = RGB(16, 185, 129);
constexpr COLORREF kColAmber = RGB(217, 119, 6);
constexpr COLORREF kColRed = RGB(220, 38, 38);
constexpr COLORREF kColSlate = RGB(51, 65, 85);
constexpr COLORREF kColDisabled = RGB(38, 44, 58);

constexpr int kBannerH = 140;

HWND g_wnd, g_game, g_elf, g_install, g_log, g_status, g_prereq_lbl;
HWND g_build, g_update, g_prereqs, g_cancel, g_play, g_saves;
HWND g_lbl_game, g_lbl_elf, g_lbl_install, g_note;
HFONT g_font_ui, g_font_bold, g_font_mono;
HBRUSH g_br_bg, g_br_card;
HICON g_game_hicon = nullptr;

std::unique_ptr<Gdiplus::Bitmap> g_banner_bmp;
std::unique_ptr<Gdiplus::Bitmap> g_icon_bmp;
path g_art_root;
std::wstring g_hero_title = L"CellForge";
std::wstring g_hero_sub = L"PS3 Static Recompiler, Installer & Vulkan Launcher";
bool g_hero_built = false;
bool g_prereqs_ok = false;

std::thread g_worker;
std::atomic<bool> g_cancel_flag{false}, g_busy{false};
path g_ini, g_exe_dir, g_repo, g_saved_work_dir;

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(std::max(0, n)), L'\0');
  if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}

std::wstring get_text(HWND h) {
  std::wstring s(size_t(GetWindowTextLengthW(h)) + 1, L'\0');
  GetWindowTextW(h, s.data(), int(s.size()));
  s.resize(wcslen(s.c_str()));
  return s;
}

void append_log(const std::wstring& line) {
  const int len = GetWindowTextLengthW(g_log);
  if (len > 900000) {
    SendMessageW(g_log, EM_SETSEL, 0, len / 2);
    SendMessageW(g_log, EM_REPLACESEL, FALSE, LPARAM(L""));
  }
  const int end = GetWindowTextLengthW(g_log);
  SendMessageW(g_log, EM_SETSEL, end, end);
  SendMessageW(g_log, EM_REPLACESEL, FALSE, LPARAM((line + L"\r\n").c_str()));
}

void set_busy(bool busy) {
  g_busy = busy;
  EnableWindow(g_build, !busy);
  EnableWindow(g_update, !busy);
  EnableWindow(g_prereqs, !busy);
  EnableWindow(g_play, !busy);
  EnableWindow(g_saves, !busy);
  EnableWindow(g_cancel, busy);
  EnableWindow(GetDlgItem(g_wnd, ID_GAME_BROWSE), !busy);
  EnableWindow(GetDlgItem(g_wnd, ID_ELF_BROWSE), !busy);
  EnableWindow(GetDlgItem(g_wnd, ID_INSTALL_BROWSE), !busy);
  SendMessageW(g_game, EM_SETREADONLY, busy, 0);
  SendMessageW(g_elf, EM_SETREADONLY, busy, 0);
  SendMessageW(g_install, EM_SETREADONLY, busy, 0);
  InvalidateRect(g_wnd, nullptr, FALSE);
}

builder::Settings current_settings() {
  builder::Settings s;
  s.game_root = get_text(g_game);
  s.elf = get_text(g_elf);
  s.install_dir = get_text(g_install);
  const builder::Validation v = builder::validate(s);
  const builder::PrereqStatus st = builder::check_prereqs(g_exe_dir);
  g_repo = st.repo_dir;
  std::error_code ec;
  if (!g_saved_work_dir.empty() && builder::fs::exists(g_saved_work_dir / "build", ec)) {
    s.work_dir = g_saved_work_dir;
  } else if (!g_repo.empty() && (v.title_id == "BLUS30464" || v.title_id == "BLES00760") &&
             builder::fs::exists(g_repo / "out" / "skate3" / "build", ec)) {
    s.work_dir = g_repo / "out" / "skate3";
  } else {
    s.work_dir = builder::default_work_dir(g_repo, v);
  }
  if (s.install_dir.empty() && v.ok) s.install_dir = builder::default_install_dir(v);
  return s;
}

void reload_artwork(const path& game_root) {
  if (game_root == g_art_root) return;
  g_art_root = game_root;
  g_banner_bmp.reset();
  g_icon_bmp.reset();
  if (g_game_hicon) {
    DestroyIcon(g_game_hicon);
    g_game_hicon = nullptr;
  }
  std::error_code ec;
  const path pic1 = game_root / "PS3_GAME" / "PIC1.PNG";
  if (builder::fs::exists(pic1, ec)) {
    auto b = std::make_unique<Gdiplus::Bitmap>(pic1.wstring().c_str());
    if (b->GetLastStatus() == Gdiplus::Ok) g_banner_bmp = std::move(b);
  }
  const path icon0 = game_root / "PS3_GAME" / "ICON0.PNG";
  if (builder::fs::exists(icon0, ec)) {
    auto b = std::make_unique<Gdiplus::Bitmap>(icon0.wstring().c_str());
    if (b->GetLastStatus() == Gdiplus::Ok) {
      b->GetHICON(&g_game_hicon);
      if (g_game_hicon && g_wnd) {
        SendMessageW(g_wnd, WM_SETICON, ICON_BIG, LPARAM(g_game_hicon));
        SendMessageW(g_wnd, WM_SETICON, ICON_SMALL, LPARAM(g_game_hicon));
      }
      g_icon_bmp = std::move(b);
    }
  }
}

std::wstring pick(bool folder, const std::wstring& current, const wchar_t* title) {
  std::wstring result;
  IFileOpenDialog* dlg = nullptr;
  if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg)))) return result;
  DWORD opts = 0;
  dlg->GetOptions(&opts);
  dlg->SetOptions(opts | FOS_FORCEFILESYSTEM | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
  dlg->SetTitle(title);
  if (!folder) {
    const COMDLG_FILTERSPEC filters[] = {{L"ELF files", L"*.elf;*.ELF"}, {L"All files", L"*.*"}};
    dlg->SetFileTypes(2, filters);
  }
  if (!current.empty()) {
    IShellItem* item = nullptr;
    const std::wstring dir = folder ? current : path(current).parent_path().wstring();
    if (SUCCEEDED(SHCreateItemFromParsingName(dir.c_str(), nullptr, IID_PPV_ARGS(&item)))) {
      dlg->SetFolder(item);
      item->Release();
    }
  }
  if (SUCCEEDED(dlg->Show(g_wnd))) {
    IShellItem* item = nullptr;
    if (SUCCEEDED(dlg->GetResult(&item))) {
      PWSTR p = nullptr;
      if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) {
        result = p;
        CoTaskMemFree(p);
      }
      item->Release();
    }
  }
  dlg->Release();
  return result;
}

void refresh_status() {
  if (g_busy) return;
  const builder::PrereqStatus st = builder::check_prereqs(g_exe_dir);
  g_repo = st.repo_dir;
  g_prereqs_ok = st.all_ready();
  std::wstring pmsg = L"Prerequisites:  Source ";
  pmsg += st.has_source ? L"[OK]" : L"[Download]";
  pmsg += L"   •   CMake ";
  pmsg += st.has_cmake ? L"[OK]" : L"[Missing]";
  pmsg += L"   •   C++ Build Tools ";
  pmsg += st.has_msvc ? L"[OK]" : L"[Missing]";
  if (!g_prereqs_ok) pmsg += L"   (click Install Prereqs or Build & Install)";
  SetWindowTextW(g_prereq_lbl, pmsg.c_str());

  builder::Settings s = current_settings();
  reload_artwork(s.game_root);
  const builder::Validation v = builder::validate(s);
  if (v.ok && get_text(g_install).empty()) {
    SetWindowTextW(g_install, builder::default_install_dir(v).wstring().c_str());
    s.install_dir = builder::default_install_dir(v);
  }
  std::error_code ec;
  const bool has_inst = v.ok && builder::fs::exists(builder::installed_exe(s, v), ec);
  const bool has_work = v.ok && builder::fs::exists(builder::game_exe(s), ec);
  g_hero_built = has_inst || has_work;

  std::wstring msg;
  if (get_text(g_game).empty() && get_text(g_elf).empty()) {
    g_hero_title = L"CellForge Installer";
    g_hero_sub = L"PS3 Static Recompiler & Standalone Game Builder";
    msg = L"Choose your game folder, decrypted ELF, and standalone install folder.";
  } else if (!v.ok) {
    g_hero_title = L"CellForge Installer";
    g_hero_sub = widen(v.error);
    msg = widen(v.error);
  } else {
    g_hero_title = widen(builder::game_name(v));
    g_hero_sub = (v.title_id.empty() ? L"Ready" : widen(v.title_id)) +
                 std::wstring(has_inst ? L"  •  Installed (Standalone)"
                              : has_work ? L"  •  Built — Ready to Play"
                                         : L"  •  Ready to Build & Install");
    msg = L"Persistent saves: " + builder::saves_dir(v).wstring();
  }
  SetWindowTextW(g_status, msg.c_str());
  RECT r{0, 0, 2000, kBannerH};
  InvalidateRect(g_wnd, &r, FALSE);
}

enum class JobKind { Build, Update, Prereqs };

void run_job(JobKind kind) {
  builder::Settings s = current_settings();
  if (kind != JobKind::Prereqs) {
    const builder::Validation v = builder::validate(s);
    if (!v.ok) {
      MessageBoxW(g_wnd, widen(v.error).c_str(), L"CellForge", MB_ICONWARNING);
      return;
    }
    builder::save_settings(g_ini, s);
  }
  SetWindowTextW(g_log, L"");
  set_busy(true);
  SetWindowTextW(g_status, kind == JobKind::Prereqs ? L"Checking & installing prerequisites..."
                            : kind == JobKind::Update ? L"Updating runtime & standalone game..."
                                                      : L"Building & installing standalone game...");
  g_cancel_flag = false;
  if (g_worker.joinable()) g_worker.join();
  g_worker = std::thread([s, kind] {
    auto log_cb = [](const std::string& line) {
      PostMessageW(g_wnd, WM_APP_LOG, 0, LPARAM(new std::string(line)));
    };
    bool ok = false;
    if (kind == JobKind::Prereqs)
      ok = builder::ensure_prereqs(g_exe_dir, log_cb, g_cancel_flag, false);
    else if (kind == JobKind::Update)
      ok = builder::update_runtime(s, g_repo, log_cb, g_cancel_flag);
    else
      ok = builder::build(s, g_repo, log_cb, g_cancel_flag);
    PostMessageW(g_wnd, WM_APP_DONE, ok, 0);
  });
}

void play() {
  builder::Settings s = current_settings();
  const builder::Validation v = builder::validate(s);
  if (!v.ok) {
    MessageBoxW(g_wnd, widen(v.error).c_str(), L"CellForge", MB_ICONWARNING);
    return;
  }
  builder::save_settings(g_ini, s);
  std::string err;
  if (!builder::launch(s, err))
    MessageBoxW(g_wnd, widen(err).c_str(), L"CellForge", MB_ICONWARNING);
  else {
    std::error_code ec;
    const path inst = builder::installed_exe(s, v);
    append_log(L"Launched " + (builder::fs::exists(inst, ec) ? inst.wstring() : builder::game_exe(s).wstring()));
  }
}

void open_saves() {
  builder::Settings s = current_settings();
  const builder::Validation v = builder::validate(s);
  const path dir = builder::saves_dir(v);
  std::error_code ec;
  builder::fs::create_directories(dir, ec);
  ShellExecuteW(g_wnd, L"open", dir.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

HWND make(const wchar_t* cls, const wchar_t* text, DWORD style, int id, HFONT font = nullptr, DWORD ex = 0) {
  HWND h = CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 10, 10, g_wnd, HMENU(INT_PTR(id)),
                           GetModuleHandleW(nullptr), nullptr);
  SendMessageW(h, WM_SETFONT, WPARAM(font ? font : g_font_ui), TRUE);
  return h;
}

COLORREF button_color(int id, bool disabled) {
  if (disabled) return kColDisabled;
  switch (id) {
    case ID_PLAY: return kColGreen;
    case ID_BUILD: return kColBlue;
    case ID_UPDATE: return kColSky;
    case ID_PREREQS: return g_prereqs_ok ? kColSlate : kColAmber;
    case ID_CANCEL: return kColRed;
    default: return kColSlate;
  }
}

void draw_button(const DRAWITEMSTRUCT* dis) {
  const bool disabled = (dis->itemState & ODS_DISABLED) != 0;
  const bool pressed = (dis->itemState & ODS_SELECTED) != 0;
  COLORREF col = button_color(int(dis->CtlID), disabled);
  if (pressed && !disabled) {
    col = RGB(std::max(0, int(GetRValue(col)) - 25),
              std::max(0, int(GetGValue(col)) - 25),
              std::max(0, int(GetBValue(col)) - 25));
  }
  Gdiplus::Graphics g(dis->hDC);
  g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
  Gdiplus::SolidBrush br(Gdiplus::Color(255, GetRValue(col), GetGValue(col), GetBValue(col)));
  const RECT& r = dis->rcItem;
  const Gdiplus::RectF rf(float(r.left), float(r.top), float(r.right - r.left), float(r.bottom - r.top));
  g.FillRectangle(&br, rf);

  SetBkMode(dis->hDC, TRANSPARENT);
  SetTextColor(dis->hDC, disabled ? kColMuted : RGB(255, 255, 255));
  SelectObject(dis->hDC, g_font_bold);
  wchar_t text[64] = {};
  GetWindowTextW(dis->hwndItem, text, 64);
  RECT tr = r;
  DrawTextW(dis->hDC, text, -1, &tr, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

void paint_banner(HDC hdc, int w) {
  Gdiplus::Graphics g(hdc);
  g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
  g.SetTextRenderingHint(Gdiplus::TextRenderingHintClearTypeGridFit);

  Gdiplus::LinearGradientBrush bg(Gdiplus::Point(0, 0), Gdiplus::Point(w, kBannerH),
                                  Gdiplus::Color(255, 22, 28, 42), Gdiplus::Color(255, 14, 18, 28));
  g.FillRectangle(&bg, 0, 0, w, kBannerH);

  if (g_banner_bmp) {
    const float bw = float(g_banner_bmp->GetWidth());
    const float bh = float(g_banner_bmp->GetHeight());
    if (bw > 0 && bh > 0) {
      const float scale = std::max(float(w) / bw, float(kBannerH) / bh);
      const float dw = bw * scale, dh = bh * scale;
      g.DrawImage(g_banner_bmp.get(), (float(w) - dw) * 0.5f, (float(kBannerH) - dh) * 0.35f, dw, dh);
    }
    Gdiplus::LinearGradientBrush scrim(Gdiplus::Point(0, 0), Gdiplus::Point(0, kBannerH),
                                       Gdiplus::Color(150, 12, 16, 24), Gdiplus::Color(235, 18, 21, 28));
    g.FillRectangle(&scrim, 0, 0, w, kBannerH);
  }

  int text_x = 20;
  if (g_icon_bmp) {
    const int ih = 92;
    const int iw = std::min(168, int(float(ih) * float(g_icon_bmp->GetWidth()) /
                                     std::max(1.0f, float(g_icon_bmp->GetHeight()))));
    const int ix = 20, iy = (kBannerH - ih) / 2;
    Gdiplus::SolidBrush shadow(Gdiplus::Color(140, 0, 0, 0));
    g.FillRectangle(&shadow, ix + 3, iy + 3, iw, ih);
    g.DrawImage(g_icon_bmp.get(), ix, iy, iw, ih);
    Gdiplus::Pen border(Gdiplus::Color(180, 148, 163, 184), 1.5f);
    g.DrawRectangle(&border, ix, iy, iw, ih);
    text_x = ix + iw + 18;
  }

  Gdiplus::FontFamily ff(L"Segoe UI");
  Gdiplus::Font title_f(&ff, 22.0f, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
  Gdiplus::Font sub_f(&ff, 13.0f, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
  Gdiplus::Font tag_f(&ff, 11.0f, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
  Gdiplus::SolidBrush white(Gdiplus::Color(255, 241, 245, 249));
  Gdiplus::SolidBrush muted(Gdiplus::Color(255, 186, 198, 214));
  Gdiplus::SolidBrush accent(g_hero_built ? Gdiplus::Color(255, 52, 211, 153) : Gdiplus::Color(255, 96, 165, 250));

  g.DrawString(L"CELLFORGE  //  PS3 STATIC RECOMPILER & STANDALONE INSTALLER", -1, &tag_f,
               Gdiplus::PointF(float(text_x), 24.0f), &accent);
  g.DrawString(g_hero_title.c_str(), -1, &title_f, Gdiplus::PointF(float(text_x), 44.0f), &white);
  g.DrawString(g_hero_sub.c_str(), -1, &sub_f, Gdiplus::PointF(float(text_x), 80.0f), &muted);

  Gdiplus::Pen div(Gdiplus::Color(255, 45, 55, 72), 1.0f);
  g.DrawLine(&div, 0, kBannerH - 1, w, kBannerH - 1);
}

void layout(int w, int h) {
  const int m = 16, row = 26, btn = 92;
  int y = kBannerH + 10;
  MoveWindow(g_prereq_lbl, m, y, w - 2 * m, 18, TRUE);
  y += 24;

  MoveWindow(g_lbl_game, m, y, w - 2 * m, 18, TRUE);
  y += 19;
  MoveWindow(g_game, m, y, w - 3 * m - btn, row, TRUE);
  MoveWindow(GetDlgItem(g_wnd, ID_GAME_BROWSE), w - m - btn, y, btn, row, TRUE);
  y += row + 8;

  MoveWindow(g_lbl_elf, m, y, w - 2 * m, 18, TRUE);
  y += 19;
  MoveWindow(g_elf, m, y, w - 3 * m - btn, row, TRUE);
  MoveWindow(GetDlgItem(g_wnd, ID_ELF_BROWSE), w - m - btn, y, btn, row, TRUE);
  y += row + 8;

  MoveWindow(g_lbl_install, m, y, w - 2 * m, 18, TRUE);
  y += 19;
  MoveWindow(g_install, m, y, w - 3 * m - btn, row, TRUE);
  MoveWindow(GetDlgItem(g_wnd, ID_INSTALL_BROWSE), w - m - btn, y, btn, row, TRUE);
  y += row + 8;

  MoveWindow(g_note, m, y, w - 2 * m, 18, TRUE);
  y += 24;

  int bx = m;
  MoveWindow(g_play, bx, y, 92, 34, TRUE); bx += 100;
  MoveWindow(g_build, bx, y, 132, 34, TRUE); bx += 140;
  MoveWindow(g_update, bx, y, 130, 34, TRUE); bx += 138;
  MoveWindow(g_prereqs, bx, y, 124, 34, TRUE); bx += 132;
  MoveWindow(g_cancel, bx, y, 80, 34, TRUE);
  MoveWindow(g_saves, w - m - 96, y, 96, 34, TRUE);
  y += 42;

  MoveWindow(g_status, m, y, w - 2 * m, 18, TRUE);
  y += 22;
  MoveWindow(g_log, m, y, w - 2 * m, std::max(80, h - y - m), TRUE);
}

LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  switch (msg) {
    case WM_CREATE: {
      g_wnd = hwnd;
      BOOL dark = TRUE;
      if (FAILED(DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark))))
        DwmSetWindowAttribute(hwnd, 19, &dark, sizeof(dark));

      g_font_ui = CreateFontW(-14, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY,
                              VARIABLE_PITCH, L"Segoe UI");
      g_font_bold = CreateFontW(-13, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY,
                                VARIABLE_PITCH, L"Segoe UI");
      g_font_mono = CreateFontW(-13, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY,
                                FIXED_PITCH, L"Consolas");
      g_br_bg = CreateSolidBrush(kColBg);
      g_br_card = CreateSolidBrush(kColCard);

      g_prereq_lbl = make(L"STATIC", L"", SS_LEFTNOWORDWRAP, ID_PREREQ_STATUS, g_font_bold);

      g_lbl_game = make(L"STATIC", L"1. PS3 Game Folder (contains PS3_GAME):", 0, 0);
      g_game = make(L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, ID_GAME);
      make(L"BUTTON", L"Browse...", BS_OWNERDRAW | WS_TABSTOP, ID_GAME_BROWSE);

      g_lbl_elf = make(L"STATIC", L"2. Decrypted ELF (produced from your own copy):", 0, 0);
      g_elf = make(L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, ID_ELF);
      make(L"BUTTON", L"Browse...", BS_OWNERDRAW | WS_TABSTOP, ID_ELF_BROWSE);

      g_lbl_install = make(L"STATIC", L"3. Save / Install Folder (detached standalone game output — no CellForge needed to play):", 0, 0);
      g_install = make(L"EDIT", L"", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, ID_INSTALL);
      make(L"BUTTON", L"Browse...", BS_OWNERDRAW | WS_TABSTOP, ID_INSTALL_BROWSE);

      g_note = make(L"STATIC", L"CellForge does not decrypt games. Encrypted EBOOT.BIN files are rejected.", 0, 0);

      g_play = make(L"BUTTON", L"Play", BS_OWNERDRAW | WS_TABSTOP, ID_PLAY, g_font_bold);
      g_build = make(L"BUTTON", L"Build & Install", BS_OWNERDRAW | WS_TABSTOP, ID_BUILD, g_font_bold);
      g_update = make(L"BUTTON", L"Update Runtime", BS_OWNERDRAW | WS_TABSTOP, ID_UPDATE, g_font_bold);
      g_prereqs = make(L"BUTTON", L"Install Prereqs", BS_OWNERDRAW | WS_TABSTOP, ID_PREREQS, g_font_bold);
      g_cancel = make(L"BUTTON", L"Cancel", BS_OWNERDRAW | WS_TABSTOP, ID_CANCEL, g_font_bold);
      g_saves = make(L"BUTTON", L"Saves", BS_OWNERDRAW | WS_TABSTOP, ID_SAVES, g_font_bold);

      g_status = make(L"STATIC", L"", SS_LEFTNOWORDWRAP, ID_STATUS);
      g_log = make(L"EDIT", L"",
                   WS_BORDER | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL | ES_READONLY,
                   ID_LOG, g_font_mono);
      SendMessageW(g_log, EM_SETLIMITTEXT, 0, 0);

      const builder::Settings s = builder::load_settings(g_ini);
      g_saved_work_dir = s.work_dir;
      SetWindowTextW(g_game, s.game_root.wstring().c_str());
      SetWindowTextW(g_elf, s.elf.wstring().c_str());
      SetWindowTextW(g_install, s.install_dir.wstring().c_str());
      set_busy(false);
      refresh_status();
      return 0;
    }
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hwnd, &ps);
      RECT rc;
      GetClientRect(hwnd, &rc);
      paint_banner(hdc, rc.right);
      RECT below = rc;
      below.top = kBannerH;
      FillRect(hdc, &below, g_br_bg);
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_ERASEBKGND:
      return 1;
    case WM_CTLCOLORSTATIC: {
      HDC hdc = HDC(wp);
      HWND ctl = HWND(lp);
      SetBkColor(hdc, kColBg);
      if (ctl == g_prereq_lbl)
        SetTextColor(hdc, g_prereqs_ok ? RGB(52, 211, 153) : RGB(251, 191, 36));
      else
        SetTextColor(hdc, (ctl == g_note || ctl == g_status) ? kColMuted : kColText);
      return LRESULT(g_br_bg);
    }
    case WM_CTLCOLOREDIT: {
      HDC hdc = HDC(wp);
      SetBkColor(hdc, kColCard);
      SetTextColor(hdc, kColText);
      return LRESULT(g_br_card);
    }
    case WM_DRAWITEM:
      draw_button(reinterpret_cast<const DRAWITEMSTRUCT*>(lp));
      return TRUE;
    case WM_SIZE:
      layout(LOWORD(lp), HIWORD(lp));
      InvalidateRect(hwnd, nullptr, FALSE);
      return 0;
    case WM_GETMINMAXINFO:
      reinterpret_cast<MINMAXINFO*>(lp)->ptMinTrackSize = {780, 580};
      return 0;
    case WM_COMMAND:
      switch (LOWORD(wp)) {
        case ID_GAME_BROWSE: {
          const std::wstring p = pick(true, get_text(g_game), L"Choose the PS3 game folder (contains PS3_GAME)");
          if (!p.empty()) SetWindowTextW(g_game, p.c_str());
          break;
        }
        case ID_ELF_BROWSE: {
          const std::wstring p = pick(false, get_text(g_elf), L"Choose the decrypted ELF produced from your copy");
          if (!p.empty()) SetWindowTextW(g_elf, p.c_str());
          break;
        }
        case ID_INSTALL_BROWSE: {
          const std::wstring p = pick(true, get_text(g_install), L"Choose where to save the standalone game folder");
          if (!p.empty()) SetWindowTextW(g_install, p.c_str());
          break;
        }
        case ID_GAME:
        case ID_ELF:
        case ID_INSTALL:
          if (HIWORD(wp) == EN_CHANGE) refresh_status();
          break;
        case ID_BUILD: run_job(JobKind::Build); break;
        case ID_UPDATE: run_job(JobKind::Update); break;
        case ID_PREREQS: run_job(JobKind::Prereqs); break;
        case ID_CANCEL:
          g_cancel_flag = true;
          SetWindowTextW(g_status, L"Cancelling...");
          break;
        case ID_PLAY: play(); break;
        case ID_SAVES: open_saves(); break;
      }
      return 0;
    case WM_APP_LOG: {
      std::unique_ptr<std::string> line(reinterpret_cast<std::string*>(lp));
      append_log(widen(*line));
      return 0;
    }
    case WM_APP_DONE:
      set_busy(false);
      refresh_status();
      if (wp) append_log(L"\r\nDone — press Play to launch your standalone game!");
      return 0;
    case WM_CLOSE:
      if (g_busy) {
        if (MessageBoxW(hwnd, L"An operation is running. Cancel it and exit?", L"CellForge", MB_YESNO | MB_ICONQUESTION) != IDYES)
          return 0;
        g_cancel_flag = true;
      }
      DestroyWindow(hwnd);
      return 0;
    case WM_DESTROY:
      if (g_worker.joinable()) g_worker.join();
      if (g_game_hicon) DestroyIcon(g_game_hicon);
      DeleteObject(g_font_ui);
      DeleteObject(g_font_bold);
      DeleteObject(g_font_mono);
      DeleteObject(g_br_bg);
      DeleteObject(g_br_card);
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int show) {
  CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
  ULONG_PTR gdi_token = 0;
  Gdiplus::GdiplusStartupInput gdi_in;
  Gdiplus::GdiplusStartup(&gdi_token, &gdi_in, nullptr);

  wchar_t exe[MAX_PATH];
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  g_exe_dir = path(exe).parent_path();
  g_ini = g_exe_dir / "cellforge.ini";
  if (!builder::fs::exists(g_ini) && builder::fs::exists(g_exe_dir / "builder.ini"))
    g_ini = g_exe_dir / "builder.ini";

  WNDCLASSEXW wc{sizeof(wc)};
  wc.lpfnWndProc = wndproc;
  wc.hInstance = inst;
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  wc.hbrBackground = CreateSolidBrush(kColBg);
  wc.lpszClassName = L"CellForgeInstaller";
  RegisterClassExW(&wc);

  HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"CellForge Installer & Launcher",
                              WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 860, 660,
                              nullptr, nullptr, inst, nullptr);
  ShowWindow(hwnd, show);
  UpdateWindow(hwnd);

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(hwnd, &msg)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }
  g_banner_bmp.reset();
  g_icon_bmp.reset();
  if (gdi_token) Gdiplus::GdiplusShutdown(gdi_token);
  CoUninitialize();
  return 0;
}
