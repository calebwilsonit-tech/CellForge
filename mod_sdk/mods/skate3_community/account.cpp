#include "account.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincrypt.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace s3c {
namespace {

std::string store_path() {
  char buf[MAX_PATH] = {};
  DWORD n = GetEnvironmentVariableA("LOCALAPPDATA", buf, MAX_PATH);
  std::string dir = n ? std::string(buf, n) : std::string(".");
  dir += "\\CellForge";
  CreateDirectoryA(dir.c_str(), nullptr);
  return dir + "\\rpcn_account.dat";
}

std::string to_hex(const BYTE* p, DWORD n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (DWORD i = 0; i < n; ++i) { s += d[p[i] >> 4]; s += d[p[i] & 15]; }
  return s;
}
bool from_hex(const std::string& s, std::vector<BYTE>& out) {
  if (s.size() % 2) return false;
  out.clear();
  for (size_t i = 0; i < s.size(); i += 2) {
    char b[3] = {s[i], s[i + 1], 0};
    char* e = nullptr;
    out.push_back(BYTE(std::strtoul(b, &e, 16)));
    if (e != b + 2) return false;
  }
  return true;
}

std::wstring widen(const std::string& s) {
  if (s.empty()) return L"";
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), w.data(), n);
  return w;
}
std::string narrow(const std::wstring& w) {
  if (w.empty()) return "";
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

// ---- a tiny modal form: message, labelled text fields, buttons --------------------------------------------------------
struct FormField {
  std::wstring label;
  bool password = false;
  std::wstring text;
  HWND edit = nullptr;
};
struct Form {
  std::wstring title, message;
  std::vector<FormField> fields;
  std::vector<std::wstring> buttons;  // first one is the default (Enter)
  int pressed = -1;
  bool done = false;
};

LRESULT CALLBACK form_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
  auto* f = reinterpret_cast<Form*>(GetWindowLongPtrW(h, GWLP_USERDATA));
  if (f) {
    if (m == WM_COMMAND) {
      const int id = LOWORD(w);
      if (id == IDCANCEL) { f->pressed = int(f->buttons.size()) - 1; f->done = true; return 0; }
      if (id >= 1000 && id < 1000 + int(f->buttons.size())) {
        f->pressed = id - 1000;
        for (auto& fld : f->fields) {
          const int n = GetWindowTextLengthW(fld.edit);
          std::wstring t(size_t(n) + 1, L'\0');
          GetWindowTextW(fld.edit, t.data(), n + 1);
          t.resize(size_t(n));
          fld.text = t;
        }
        f->done = true;
        return 0;
      }
    }
    if (m == WM_CLOSE) { f->pressed = int(f->buttons.size()) - 1; f->done = true; return 0; }
  }
  return DefWindowProcW(h, m, w, l);
}

int run_form(Form& f) {
  static const wchar_t* kClass = L"CellForgeCommunityForm";
  static const bool registered = [] {
    WNDCLASSW wc{};
    wc.lpfnWndProc = form_proc;
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

  const int W = 480;
  const int msg_h = f.message.empty() ? 0 : 66;
  const int H = 56 + msg_h + int(f.fields.size()) * 46 + 64;
  HWND owner = GetForegroundWindow();
  RECT r{};
  if (!(owner && GetWindowRect(owner, &r))) r = RECT{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
  const int x = r.left + ((r.right - r.left) - W) / 2, y = r.top + ((r.bottom - r.top) - H) / 2;
  HWND h = CreateWindowExW(WS_EX_TOPMOST | WS_EX_DLGMODALFRAME | WS_EX_CONTROLPARENT, kClass, f.title.c_str(),
                           WS_POPUP | WS_CAPTION | WS_SYSMENU, x, y, W, H, owner, nullptr, GetModuleHandleW(nullptr), nullptr);
  if (!h) return -1;
  SetWindowLongPtrW(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&f));
  auto mk = [&](const wchar_t* cls, const wchar_t* text, DWORD style, int cx, int cy, int cw, int ch, int id, DWORD ex = 0) {
    HWND c = CreateWindowExW(ex, cls, text, WS_CHILD | WS_VISIBLE | style, cx, cy, cw, ch, h,
                             reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), GetModuleHandleW(nullptr), nullptr);
    SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    return c;
  };
  int cy = 12;
  if (msg_h) { mk(L"STATIC", f.message.c_str(), SS_LEFT, 16, cy, W - 48, msg_h - 6, 100); cy += msg_h; }
  int idc = 200;
  HWND first = nullptr;
  for (auto& fld : f.fields) {
    mk(L"STATIC", fld.label.c_str(), SS_LEFT, 16, cy, W - 48, 18, idc++);
    fld.edit = mk(L"EDIT", fld.text.c_str(), WS_TABSTOP | ES_AUTOHSCROLL | (fld.password ? ES_PASSWORD : 0), 16, cy + 20, W - 48, 24, idc++,
                  WS_EX_CLIENTEDGE);
    if (!first) first = fld.edit;
    cy += 46;
  }
  const int bw = 130, gap = 8;
  int bx = W - 32 - int(f.buttons.size()) * (bw + gap) + gap;
  for (size_t i = 0; i < f.buttons.size(); ++i) {
    mk(L"BUTTON", f.buttons[i].c_str(), WS_TABSTOP | (i == 0 ? BS_DEFPUSHBUTTON : BS_PUSHBUTTON), bx, cy + 8, bw, 30, 1000 + int(i));
    bx += bw + gap;
  }
  ShowWindow(h, SW_SHOW);
  SetForegroundWindow(h);
  if (first) { SetFocus(first); SendMessageW(first, EM_SETSEL, 0, -1); }
  MSG msg;
  while (!f.done && GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(h, &msg)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
  }
  DestroyWindow(h);
  DeleteObject(font);
  if (owner) SetForegroundWindow(owner);
  return f.pressed;
}

}  // namespace

bool load_credentials(Credentials& out) {
  FILE* fp = std::fopen(store_path().c_str(), "rb");
  if (!fp) return false;
  char line[8192];
  std::string l1, l2, l3;
  auto rd = [&](std::string& s) {
    if (!std::fgets(line, sizeof line, fp)) return false;
    s = line;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return true;
  };
  const bool ok = rd(l1) && rd(l2) && rd(l3) && l1 == "v1";
  std::fclose(fp);
  if (!ok) return false;
  std::vector<BYTE> blob;
  if (!from_hex(l3, blob) || blob.empty()) return false;
  DATA_BLOB in{DWORD(blob.size()), blob.data()}, plain{};
  if (!CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr, 0, &plain)) return false;
  out.npid = l2;
  const std::string joined(reinterpret_cast<char*>(plain.pbData), plain.cbData);
  const size_t sep = joined.find('\x01');
  out.password = joined.substr(0, sep);
  out.token = sep == std::string::npos ? std::string() : joined.substr(sep + 1);
  SecureZeroMemory(plain.pbData, plain.cbData);
  LocalFree(plain.pbData);
  return !out.npid.empty() && !out.password.empty();
}

void save_credentials(const Credentials& c) {
  std::string joined = c.password;
  joined += '\x01';
  joined += c.token;
  DATA_BLOB in{DWORD(joined.size()), reinterpret_cast<BYTE*>(joined.data())}, enc{};
  if (!CryptProtectData(&in, L"CellForge RPCN password", nullptr, nullptr, nullptr, 0, &enc)) return;
  if (FILE* fp = std::fopen(store_path().c_str(), "wb")) {
    std::fprintf(fp, "v1\n%s\n%s\n", c.npid.c_str(), to_hex(enc.pbData, enc.cbData).c_str());
    std::fclose(fp);
  }
  LocalFree(enc.pbData);
}

void forget_credentials() { DeleteFileA(store_path().c_str()); }

SignInAction prompt_signin(SignInForm& form) {
  Form f;
  f.title = L"Skate 3 online - sign in";
  f.message = widen(form.message.empty()
                        ? "Sign in with your RPCN account to play Skate 3 online. RPCN is the free account service used by "
                          "RPCS3. Your password is stored encrypted on this PC only."
                        : form.message);
  f.fields = {{L"RPCN username", false, widen(form.npid), nullptr},
              {L"Password", true, widen(form.password), nullptr},
              {L"Email token (only needed the first time after creating an account)", false, widen(form.token), nullptr}};
  f.buttons = {L"Sign in", L"Create account", L"Not now"};
  const int pressed = run_form(f);
  if (pressed == 0) {
    form.npid = narrow(f.fields[0].text);
    form.password = narrow(f.fields[1].text);
    form.token = narrow(f.fields[2].text);
    return SignInAction::SignIn;
  }
  if (pressed == 1) {
    form.npid = narrow(f.fields[0].text);
    form.password = narrow(f.fields[1].text);
    return SignInAction::CreateAccount;
  }
  return SignInAction::NotNow;
}

bool prompt_create_account(SignInForm& form) {
  Form f;
  f.title = L"Skate 3 online - create an RPCN account";
  f.message = widen(form.message.empty() ? "Create a free RPCN account. The name must be 3-16 letters, digits, - or _. RPCN will "
                                           "email you a verification token; enter it the first time you sign in."
                                         : form.message);
  f.fields = {{L"Username", false, widen(form.npid), nullptr},
              {L"Password", true, widen(form.password), nullptr},
              {L"Email address", false, widen(form.email), nullptr}};
  f.buttons = {L"Create", L"Back"};
  const int pressed = run_form(f);
  form.npid = narrow(f.fields[0].text);
  form.password = narrow(f.fields[1].text);
  form.email = narrow(f.fields[2].text);
  return pressed == 0;
}

}  // namespace s3c
