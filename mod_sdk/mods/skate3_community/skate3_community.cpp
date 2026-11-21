// Skate 3 community online (mod) — milestone 1: real networking for the PS3 network stack.
//
// Skate 3's EA servers are gone, but the community runs a replacement "Blaze" server. The game reaches it
// through the PS3 network libraries (sys_net sockets, cellNetCtl, sceNp). CellForge's built-in versions report
// "offline"; this mod replaces them so the game can connect:
//
//   * sys_net      -> Windows sockets (Winsock), with the EA hostnames redirected to the community server
//   * cellNetCtl   -> reports a connected network with the real local address
//   * sceNp        -> reports a signed-in, online account
//
// Everything is automatic: drop the DLL (and optionally skate3_community.ini) into mods/, nothing else to set up.
// Remove the DLL (or set enabled=0) and the game goes back to offline mode.

#include "cellforge_mod.h"
#include "account.h"
#include "rpcn.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <utility>
#include <chrono>
#include <thread>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using ppu::Context;
using cellforge::read_u8;
using cellforge::read_u16;
using cellforge::read_u32;
using cellforge::write_u8;
using cellforge::write_u16;
using cellforge::write_u32;

// ---- configuration -------------------------------------------------------------------------------------------
struct Settings {
  bool enabled = false;   // experimental: off unless the ini says enabled=1
  bool trace = false;                          // log every network call
  bool flow_trace = false;                     // log the game's sign-in state machine (Skate 3 only; for troubleshooting)
  std::string server_ip = "172.237.109.212";   // community Blaze server (public)
  std::vector<std::string> redirect_hosts = {"gosredirector.ea.com", "downloads.skate*.online.ea.com"};
  std::string np_name = "CellForgeUser";       // online id shown to the game until a real account is signed in
  std::string rpcn_host = "np.rpcs3.net";      // RPCN account service (the one RPCS3 uses)
  int rpcn_port = 31313;
  bool prompt_signin = true;                   // ask for an RPCN login in-game when none is saved
  bool webkit_autoquit = true;                 // Skate 3: end the front end's browser state as if the page had closed
  bool ws_ok = true;                           // treat "404" from the community server's missing /skate3/ws/ web services as success
  std::string webkit_mode = "skip";            // "skip": never start the game's embedded browser; "standin": inert fake browser
  bool dev_fake_signin = false;                // developer testing: pretend to be signed in with a dummy ticket (no RPCN)
} g_cfg;

std::string lower(std::string s) {
  for (char& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find_first_of(",;| ", i);
    if (j == std::string::npos) j = s.size();
    if (j > i) out.push_back(lower(s.substr(i, j - i)));
    i = j + 1;
  }
  return out;
}

bool g_webkit_credits = false;  // (experimental: dialog shows blank text) Skate 3: show the community server's credits text in the game's own dialog when a web page is opened
bool g_webkit_skip_prompt = true;  // Skate 3: skip the browser "exit?" prompt (see FlowWrap<0x0040BA04>)
void load_settings() {
  HMODULE self = nullptr;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&load_settings), &self);
  char path[MAX_PATH] = {};
  GetModuleFileNameA(self, path, MAX_PATH);
  std::string ini = path;
  const size_t dot = ini.find_last_of('.');
  if (dot != std::string::npos) ini.resize(dot);
  ini += ".ini";
  cellforge::Config cfg;
  if (!cfg.load(ini)) {
    cellforge::logf("skate3_community: no %s, using defaults", ini.c_str());
    return;
  }
  g_cfg.enabled = cfg.get_bool("enabled", false);
  g_cfg.trace = cfg.get_bool("trace", false);
  g_cfg.flow_trace = cfg.get_bool("flow_trace", false);
  g_cfg.server_ip = cfg.get_string("server_ip", g_cfg.server_ip);
  const std::string hosts = cfg.get_string("redirect_hosts", "");
  if (!hosts.empty()) g_cfg.redirect_hosts = split_list(hosts);
  g_cfg.np_name = cfg.get_string("np_name", g_cfg.np_name);
  g_cfg.rpcn_host = cfg.get_string("rpcn_host", g_cfg.rpcn_host);
  g_cfg.rpcn_port = cfg.get_int("rpcn_port", g_cfg.rpcn_port);
  g_cfg.prompt_signin = cfg.get_bool("prompt_signin", true);
  g_cfg.dev_fake_signin = cfg.get_bool("dev_fake_signin", false);
  g_cfg.webkit_mode = cfg.get_string("webkit_mode", g_cfg.webkit_mode);
  g_cfg.ws_ok = cfg.get_bool("fake_webservice_ok", true);
  g_cfg.webkit_autoquit = cfg.get_bool("webkit_autoquit", true);
  g_webkit_credits = cfg.get_bool("webkit_credits", false);
  g_webkit_skip_prompt = cfg.get_bool("webkit_skip_prompt", true) && !g_webkit_credits;
}

// ---- small helpers ------------------------------------------------------------------------------------------
inline uint32_t A(Context& c, int i) { return uint32_t(c.r[3 + i]); }
inline void RET(Context& c, int64_t v) { c.r[3] = uint64_t(v); }
char* G(uint32_t ea) { return reinterpret_cast<char*>(cellforge::guest_base() + ea); }

void tracef(const char* fmt, ...) {
  if (!g_cfg.trace) return;
  char b[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  cellforge::log(b);
}

// PS3 libnet errno values (BSD numbering: EAGAIN is 35).
constexpr int kEPERM = 1, kEBADF = 9, kEFAULT = 14, kEINVAL = 22, kEAGAIN = 35, kEINPROGRESS = 36;
uint32_t g_errno_addr = 0;
void set_errno(int e) { if (g_errno_addr) cellforge::write_i32(g_errno_addr, e); }

int map_wsa_error(int e) {
  switch (e) {
    case WSAEINTR: return 4;
    case WSAEBADF: return 9;
    case WSAEACCES: return 13;
    case WSAEFAULT: return 14;
    case WSAEINVAL: return 22;
    case WSAEMFILE: return 24;
    case WSAEWOULDBLOCK: return 35;
    case WSAEINPROGRESS: return 36;
    case WSAEALREADY: return 37;
    case WSAENOTSOCK: return 38;
    case WSAEDESTADDRREQ: return 39;
    case WSAEMSGSIZE: return 40;
    case WSAEPROTOTYPE: return 41;
    case WSAENOPROTOOPT: return 42;
    case WSAEPROTONOSUPPORT: return 43;
    case WSAEOPNOTSUPP: return 45;
    case WSAEPFNOSUPPORT: return 46;
    case WSAEAFNOSUPPORT: return 47;
    case WSAEADDRINUSE: return 48;
    case WSAEADDRNOTAVAIL: return 49;
    case WSAENETDOWN: return 50;
    case WSAENETUNREACH: return 51;
    case WSAECONNABORTED: return 53;
    case WSAECONNRESET: return 54;
    case WSAENOBUFS: return 55;
    case WSAEISCONN: return 56;
    case WSAENOTCONN: return 57;
    case WSAESHUTDOWN: return 58;
    case WSAETIMEDOUT: return 60;
    case WSAECONNREFUSED: return 61;
    case WSAEHOSTDOWN: return 64;
    case WSAEHOSTUNREACH: return 65;
    default: return kEINVAL;
  }
}

void fail_wsa(Context& c, int override_errno = 0) {
  set_errno(override_errno ? override_errno : map_wsa_error(WSAGetLastError()));
  RET(c, -1);
}
void fail_errno(Context& c, int e) {
  set_errno(e);
  RET(c, -1);
}

// ---- socket table ---------------------------------------------------------------------------------------------
struct Sock {
  SOCKET s = INVALID_SOCKET;
  bool nonblock = false;
  int type = 0;
};
std::mutex g_sock_mu;
std::unordered_map<int, Sock> g_socks;
int g_next_fd = 1;

bool get_sock(int fd, Sock& out) {
  std::lock_guard<std::mutex> lk(g_sock_mu);
  auto it = g_socks.find(fd);
  if (it == g_socks.end()) return false;
  out = it->second;
  return true;
}

void set_nonblock(int fd, SOCKET s, bool nb) {
  u_long v = nb ? 1 : 0;
  ioctlsocket(s, FIONBIO, &v);
  std::lock_guard<std::mutex> lk(g_sock_mu);
  auto it = g_socks.find(fd);
  if (it != g_socks.end()) it->second.nonblock = nb;
}

// sockaddr_in as the game sees it: u8 len, u8 family, be16 port, be32 addr, 8 zero bytes.
bool read_sockaddr(uint32_t ea, sockaddr_in& out) {
  if (!ea) return false;
  out = {};
  out.sin_family = AF_INET;
  out.sin_port = htons(read_u16(ea + 2));
  out.sin_addr.s_addr = htonl(read_u32(ea + 4));
  return true;
}
void write_sockaddr(uint32_t ea, const sockaddr_in& sa) {
  if (!ea) return;
  write_u8(ea, 16);
  write_u8(ea + 1, 2);
  write_u16(ea + 2, ntohs(sa.sin_port));
  write_u32(ea + 4, ntohl(sa.sin_addr.s_addr));
  write_u32(ea + 8, 0);
  write_u32(ea + 12, 0);
}
std::string ip_string(const sockaddr_in& sa) {
  char b[32];
  inet_ntop(AF_INET, &sa.sin_addr, b, sizeof b);
  return std::string(b) + ":" + std::to_string(ntohs(sa.sin_port));
}

int to_wsa_flags(uint32_t f) {
  int w = 0;
  if (f & 0x1) w |= MSG_OOB;
  if (f & 0x2) w |= MSG_PEEK;
  if (f & 0x40) w |= MSG_WAITALL;
  return w;
}

// ---- sys_net ----------------------------------------------------------------------------------------------------
std::atomic<bool> g_wsa_up{false};
void ensure_wsa() {
  if (!g_wsa_up.exchange(true)) {
    WSADATA wd;
    WSAStartup(MAKEWORD(2, 2), &wd);
  }
}

void h_initialize_network_ex(Context& c) { ensure_wsa(); tracef("sys_net_initialize_network_ex"); RET(c, 0); }
void h_finalize_network(Context& c) { RET(c, 0); }
void h_errno_loc(Context& c) { RET(c, g_errno_addr); }
void h_ok0(Context& c) { RET(c, 0); }

void h_socket(Context& c) {
  ensure_wsa();
  const uint32_t family = A(c, 0), type = A(c, 1);
  if (family != 2) return fail_errno(c, 47);
  const int st = (type == 1 || type == 10) ? SOCK_STREAM : (type == 2 || type == 6) ? SOCK_DGRAM : 0;
  if (!st) return fail_errno(c, 43);
  SOCKET s = ::socket(AF_INET, st, 0);
  if (s == INVALID_SOCKET) return fail_wsa(c);
  int fd;
  {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    fd = g_next_fd++;
    g_socks[fd] = Sock{s, false, st};
  }
  tracef("socket(type %u) = %d", type, fd);
  RET(c, fd);
}

// The community server hosts the game's config files but none of EA's old "/skate3/ws/*.asmx" profile web services, so every
// call to them is answered with "404 Not Found". The game's sign-in waits on the profile call's result, so for those calls (and
// only those) the status line is rewritten to "200 OK" (same length) as the reply is read back.
struct HttpFix {
  std::string head;     // first bytes the game sent (the request line)
  bool ws = false;      // this connection is a /skate3/ws/ call
  std::string pending;  // reply bytes read from the server but not yet handed to the game
};
std::mutex g_http_mu;
std::map<int, HttpFix> g_http;

void trace_flush_fd(int fd);
void h_socketclose(Context& c) {
  const int fd = int(A(c, 0));
  Sock sk;
  {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    auto it = g_socks.find(fd);
    if (it == g_socks.end()) return fail_errno(c, kEBADF);
    sk = it->second;
    g_socks.erase(it);
  }
  closesocket(sk.s);
  trace_flush_fd(fd);
  {
    std::lock_guard<std::mutex> lk(g_http_mu);
    auto it = g_http.find(fd);
    if (it != g_http.end()) {
      // Troubleshooting: after the second sign-in "StartLoginProcess" call finishes, log what the game does next.
      static std::atomic<int> slp{0};
      if (g_cfg.flow_trace && it->second.head.find("StartLoginProcess") != std::string::npos && ++slp == 2)
        cellforge::trace_syscalls(1500);
      g_http.erase(it);
    }
  }
  tracef("socketclose(%d)", fd);
  RET(c, 0);
}

void h_shutdown(Context& c) {
  Sock sk;
  if (!get_sock(int(A(c, 0)), sk)) return fail_errno(c, kEBADF);
  const int how = int(A(c, 1));
  RET(c, ::shutdown(sk.s, how == 0 ? SD_RECEIVE : how == 1 ? SD_SEND : SD_BOTH) == 0 ? 0 : -1);
  if (int64_t(c.r[3]) < 0) set_errno(map_wsa_error(WSAGetLastError()));
}

void h_connect(Context& c) {
  Sock sk;
  sockaddr_in sa;
  if (!get_sock(int(A(c, 0)), sk)) return fail_errno(c, kEBADF);
  if (!read_sockaddr(A(c, 1), sa)) return fail_errno(c, kEFAULT);
  tracef("connect(%d, %s)", int(A(c, 0)), ip_string(sa).c_str());
  if (::connect(sk.s, reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0) return RET(c, 0);
  const int e = WSAGetLastError();
  if (e == WSAEWOULDBLOCK) return fail_errno(c, kEINPROGRESS);
  cellforge::logf("skate3_community: connect %s failed (WSA %d)", ip_string(sa).c_str(), e);
  fail_wsa(c);
}

void h_bind(Context& c) {
  Sock sk;
  sockaddr_in sa;
  if (!get_sock(int(A(c, 0)), sk)) return fail_errno(c, kEBADF);
  if (!read_sockaddr(A(c, 1), sa)) return fail_errno(c, kEFAULT);
  tracef("bind(%d, %s)", int(A(c, 0)), ip_string(sa).c_str());
  if (::bind(sk.s, reinterpret_cast<sockaddr*>(&sa), sizeof sa) == 0) return RET(c, 0);
  fail_wsa(c);
}

void h_listen(Context& c) {
  Sock sk;
  if (!get_sock(int(A(c, 0)), sk)) return fail_errno(c, kEBADF);
  if (::listen(sk.s, int(A(c, 1))) == 0) return RET(c, 0);
  fail_wsa(c);
}

void h_accept(Context& c) {
  Sock sk;
  if (!get_sock(int(A(c, 0)), sk)) return fail_errno(c, kEBADF);
  sockaddr_in sa{};
  int len = sizeof sa;
  SOCKET n = ::accept(sk.s, reinterpret_cast<sockaddr*>(&sa), &len);
  if (n == INVALID_SOCKET) return fail_wsa(c);
  int fd;
  {
    std::lock_guard<std::mutex> lk(g_sock_mu);
    fd = g_next_fd++;
    g_socks[fd] = Sock{n, false, SOCK_STREAM};
  }
  if (A(c, 1)) write_sockaddr(A(c, 1), sa);
  if (A(c, 2)) write_u32(A(c, 2), 16);
  RET(c, fd);
}

void name_common(Context& c, bool peer) {
  Sock sk;
  if (!get_sock(int(A(c, 0)), sk)) return fail_errno(c, kEBADF);
  sockaddr_in sa{};
  int len = sizeof sa;
  const int rc = peer ? ::getpeername(sk.s, reinterpret_cast<sockaddr*>(&sa), &len)
                      : ::getsockname(sk.s, reinterpret_cast<sockaddr*>(&sa), &len);
  if (rc != 0) return fail_wsa(c);
  write_sockaddr(A(c, 1), sa);
  if (A(c, 2)) write_u32(A(c, 2), 16);
  RET(c, 0);
}
void h_getsockname(Context& c) { name_common(c, false); }
void h_getpeername(Context& c) { name_common(c, true); }

// Runs `fn` on the socket with MSG_DONTWAIT semantics: a blocking socket is switched to non-blocking for the call.
template <class F>
void with_dontwait(int fd, const Sock& sk, uint32_t flags, F&& fn) {
  const bool temp = (flags & 0x80) && !sk.nonblock;
  if (temp) { u_long v = 1; ioctlsocket(sk.s, FIONBIO, &v); }
  fn();
  if (temp) { u_long v = 0; ioctlsocket(sk.s, FIONBIO, &v); }
  (void)fd;
}

// Trace helper: collects each socket's traffic per direction into text lines (printable ASCII; '.' for the rest, '|' for
// newlines) and logs a line when it ends, grows long or the direction changes, so byte-at-a-time HTTP reads stay readable.
struct TraceBuf { std::string text; std::string hex; const char* dir = ""; int bytes = 0; int nonprint = 0; };
std::mutex g_trace_mu;
std::map<int, TraceBuf> g_trace_bufs;
void trace_flush_locked(int fd, TraceBuf& tb) {
  if (tb.bytes) {
    // Mostly-binary traffic (the Blaze protocol) also gets its first bytes in hex so frame headers can be decoded.
    if (tb.nonprint * 3 > tb.bytes)
      cellforge::logf("skate3_community: %s(%d) %d bytes: hex %s | %s", tb.dir, fd, tb.bytes, tb.hex.c_str(), tb.text.c_str());
    else
      cellforge::logf("skate3_community: %s(%d) %d bytes: %s", tb.dir, fd, tb.bytes, tb.text.c_str());
  }
  tb.text.clear();
  tb.hex.clear();
  tb.bytes = 0;
  tb.nonprint = 0;
}
void trace_flush_fd(int fd) {
  std::lock_guard<std::mutex> lk(g_trace_mu);
  auto it = g_trace_bufs.find(fd);
  if (it != g_trace_bufs.end()) { trace_flush_locked(fd, it->second); g_trace_bufs.erase(it); }
}
void trace_bytes(const char* what, int fd, uint32_t addr, int shown, int result) {
  if (!g_cfg.trace) return;
  if (result < 0) return trace_flush_fd(fd);  // would-block etc.: show what was collected so far
  std::lock_guard<std::mutex> lk(g_trace_mu);
  TraceBuf& tb = g_trace_bufs[fd];
  if (tb.bytes && tb.dir != what) trace_flush_locked(fd, tb);
  tb.dir = what;
  for (int i = 0; i < shown; ++i) {
    const uint8_t b = read_u8(addr + uint32_t(i));
    ++tb.bytes;
    const bool printable = (b >= 32 && b < 127);
    if (!printable && b != 10) ++tb.nonprint;
    if (tb.text.size() < 400) tb.text += printable ? char(b) : (b == 10 ? '|' : '.');
    if (tb.hex.size() < 96) { char h[4]; std::snprintf(h, sizeof h, "%02X", b); tb.hex += h; }
    if (b == 10 || tb.text.size() >= 400) trace_flush_locked(fd, tb);
  }
}

void h_send(Context& c) {
  const int fd = int(A(c, 0));
  Sock sk;
  if (!get_sock(fd, sk)) return fail_errno(c, kEBADF);
  int n = 0;
  with_dontwait(fd, sk, A(c, 3), [&] { n = ::send(sk.s, G(A(c, 1)), int(A(c, 2)), to_wsa_flags(A(c, 3))); });
  if (g_cfg.trace) trace_bytes("send", fd, A(c, 1), n < 0 ? 0 : n, n);
  if (n < 0) return fail_wsa(c);
  if ((g_cfg.ws_ok || g_cfg.flow_trace) && n > 0) {
    std::lock_guard<std::mutex> lk(g_http_mu);
    HttpFix& hf = g_http[fd];
    if (hf.head.size() < 160) {
      for (int i = 0; i < n && hf.head.size() < 160; ++i) hf.head += char(read_u8(A(c, 1) + uint32_t(i)));
      if (hf.head.find("/skate3/ws/") != std::string::npos) hf.ws = true;
    }
    // Troubleshooting: when the second sign-in "StartLoginProcess" body goes out, log the syscalls that follow.
    if (g_cfg.flow_trace && hf.head.find("StartLoginProcess") != std::string::npos && n >= 11 && read_u8(A(c, 1)) == 0x70 &&
        read_u8(A(c, 1) + 1) == 0x6C) {  // body starts with "pl" (platformId=...)
      static std::atomic<int> bodies{0};
      if (++bodies == 2) cellforge::trace_syscalls(6000);
    }
  }
  RET(c, n);
}

// What EA's web services answered (plain XML, no encoding attribute: the game cannot take one). Empty = leave the reply alone.
std::string ws_reply_body(const std::string& request_head) {
  auto is = [&](const char* name) { return request_head.find(std::string("/") + name) != std::string::npos; };
  const std::string xml = "<?xml version=\"1.0\"?>";
  if (is("StartLoginProcess"))
    return xml + "<LoginInfoContainer><privacyFlagContainer /><awardedBoardSales>0</awardedBoardSales><teamInfo><teamId>0</teamId>"
                 "<numMembers>0</numMembers></teamInfo></LoginInfoContainer>";
  if (is("SetUserDLC")) return xml + "<BoolContainer><value>true</value></BoolContainer>";
  if (is("AddAchievement") || is("SetUserAchievements")) return xml + "<IntegerContainer><value>0</value></IntegerContainer>";
  if (is("UploadAIProfile")) return xml + "<LongContainer><value>2</value></LongContainer>";
  return std::string();
}

void h_recv(Context& c) {
  const int fd = int(A(c, 0));
  Sock sk;
  if (!get_sock(fd, sk)) return fail_errno(c, kEBADF);
  int n = 0;
  HttpFix* hf = nullptr;
  if (g_cfg.ws_ok) {
    std::lock_guard<std::mutex> lk(g_http_mu);
    auto it = g_http.find(fd);
    if (it != g_http.end() && it->second.ws) hf = &it->second;
  }
  if (hf) {
    if (hf->pending.empty()) {
      char tmp[4096];
      int got = 0;
      with_dontwait(fd, sk, A(c, 3), [&] { got = ::recv(sk.s, tmp, sizeof tmp, to_wsa_flags(A(c, 3))); });
      if (got > 0) {
        hf->pending.assign(tmp, size_t(got));
        const size_t pos = hf->pending.find("404 Not Found");
        const std::string body = pos != std::string::npos ? ws_reply_body(hf->head) : std::string();
        if (!body.empty()) {
          // The community server has none of EA's profile web services. For the calls whose reply format is known, the reply
          // is replaced by what EA's server would have sent, so the game's own screens get the data they expect.
          hf->pending = "HTTP/1.1 200 OK\r\nServer: Microsoft-IIS/6.0\r\nContent-Type: text/xml\r\nContent-Length: " +
                        std::to_string(body.size()) + "\r\n\r\n" + body;
          tracef("fd %d: web service reply replaced (%u byte body)", fd, unsigned(body.size()));
        }
      } else {
        n = got;
      }
    }
    if (!hf->pending.empty()) {
      n = int(std::min<size_t>(hf->pending.size(), A(c, 2)));
      for (int i = 0; i < n; ++i) cellforge::write_u8(A(c, 1) + uint32_t(i), uint8_t(hf->pending[size_t(i)]));
      hf->pending.erase(0, size_t(n));
    }
  } else {
    with_dontwait(fd, sk, A(c, 3), [&] { n = ::recv(sk.s, G(A(c, 1)), int(A(c, 2)), to_wsa_flags(A(c, 3))); });
  }
  if (g_cfg.trace) trace_bytes("recv", fd, A(c, 1), n < 0 ? 0 : n, n < 0 ? -WSAGetLastError() : n);
  if (n < 0) return fail_wsa(c);
  RET(c, n);
}

void h_sendto(Context& c) {  // (fd, buf, len, flags, sockaddr*, tolen)
  const int fd = int(A(c, 0));
  Sock sk;
  if (!get_sock(fd, sk)) return fail_errno(c, kEBADF);
  int n = 0;
  sockaddr_in sa;
  const bool has_to = read_sockaddr(A(c, 4), sa);
  with_dontwait(fd, sk, A(c, 3), [&] {
    n = has_to ? ::sendto(sk.s, G(A(c, 1)), int(A(c, 2)), to_wsa_flags(A(c, 3)), reinterpret_cast<sockaddr*>(&sa), sizeof sa)
               : ::send(sk.s, G(A(c, 1)), int(A(c, 2)), to_wsa_flags(A(c, 3)));
  });
  if (n < 0) return fail_wsa(c);
  RET(c, n);
}

void h_recvfrom(Context& c) {  // (fd, buf, len, flags, sockaddr*, socklen*)
  const int fd = int(A(c, 0));
  Sock sk;
  if (!get_sock(fd, sk)) return fail_errno(c, kEBADF);
  int n = 0;
  sockaddr_in sa{};
  int salen = sizeof sa;
  with_dontwait(fd, sk, A(c, 3), [&] {
    n = ::recvfrom(sk.s, G(A(c, 1)), int(A(c, 2)), to_wsa_flags(A(c, 3)), reinterpret_cast<sockaddr*>(&sa), &salen);
  });
  if (n < 0) {
    // A too-large datagram still delivers the (truncated) data on the PS3.
    if (WSAGetLastError() != WSAEMSGSIZE) return fail_wsa(c);
    n = int(A(c, 2));
  }
  if (A(c, 4)) write_sockaddr(A(c, 4), sa);
  if (A(c, 5)) write_u32(A(c, 5), 16);
  RET(c, n);
}

// PS3 socket option numbers.
constexpr uint32_t kSolSocket = 0xFFFF, kIpprotoTcp = 6;
void h_setsockopt(Context& c) {  // (fd, level, optname, optval*, optlen)
  const int fd = int(A(c, 0));
  Sock sk;
  if (!get_sock(fd, sk)) return fail_errno(c, kEBADF);
  const uint32_t level = A(c, 1), opt = A(c, 2), val = A(c, 3), len = A(c, 4);
  int ival = (val && len >= 4) ? int(read_u32(val)) : 0;
  tracef("setsockopt(%d, 0x%X, 0x%X, %d, len %u)", fd, level, opt, ival, len);
  if (level == kSolSocket) {
    switch (opt) {
      case 0x1100: set_nonblock(fd, sk.s, ival != 0); return RET(c, 0);                            // SO_NBIO
      case 0x0004: setsockopt(sk.s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&ival), 4); return RET(c, 0);
      case 0x0008: setsockopt(sk.s, SOL_SOCKET, SO_KEEPALIVE, reinterpret_cast<char*>(&ival), 4); return RET(c, 0);
      case 0x0020: setsockopt(sk.s, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<char*>(&ival), 4); return RET(c, 0);
      case 0x1001: setsockopt(sk.s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char*>(&ival), 4); return RET(c, 0);
      case 0x1002: setsockopt(sk.s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<char*>(&ival), 4); return RET(c, 0);
      case 0x1005:
      case 0x1006: {  // SO_SNDTIMEO / SO_RCVTIMEO: be s64 sec + be s64 usec, or a u32 of microseconds
        uint64_t us = 0;
        if (len >= 16) us = cellforge::read_u64(val) * 1000000ull + cellforge::read_u64(val + 8);
        else if (len >= 4) us = uint32_t(ival);
        DWORD ms = DWORD(us / 1000);
        setsockopt(sk.s, SOL_SOCKET, opt == 0x1005 ? SO_SNDTIMEO : SO_RCVTIMEO, reinterpret_cast<char*>(&ms), sizeof ms);
        return RET(c, 0);
      }
      default: return RET(c, 0);  // SO_LINGER, SO_REUSEPORT, SO_TPPOLICY, ...: accepted and ignored
    }
  }
  if (level == kIpprotoTcp && opt == 1) {
    setsockopt(sk.s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&ival), 4);
    return RET(c, 0);
  }
  RET(c, 0);
}

void h_getsockopt(Context& c) {  // (fd, level, optname, optval*, optlen*)
  const int fd = int(A(c, 0));
  Sock sk;
  if (!get_sock(fd, sk)) return fail_errno(c, kEBADF);
  const uint32_t level = A(c, 1), opt = A(c, 2), val = A(c, 3), lenp = A(c, 4);
  int out = 0;
  if (level == kSolSocket) {
    switch (opt) {
      case 0x1007: {  // SO_ERROR
        int e = 0, l = 4;
        getsockopt(sk.s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&e), &l);
        out = e ? map_wsa_error(e) : 0;
        break;
      }
      case 0x1008: out = sk.type == SOCK_STREAM ? 1 : 2; break;  // SO_TYPE
      case 0x1100: out = sk.nonblock ? 1 : 0; break;             // SO_NBIO
      case 0x1001: { int l = 4; getsockopt(sk.s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char*>(&out), &l); break; }
      case 0x1002: { int l = 4; getsockopt(sk.s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<char*>(&out), &l); break; }
      default: break;
    }
  }
  if (val) write_u32(val, uint32_t(out));
  if (lenp) write_u32(lenp, 4);
  RET(c, 0);
}

// socketpoll(pollfd* fds, nfds, timeout): pollfd = { be s32 fd; be s16 events; be s16 revents }.
std::atomic<int> g_poll_logged{0};
void h_socketpoll(Context& c) {
  ensure_wsa();
  const uint32_t fds = A(c, 0);
  const int nfds = int(A(c, 1));
  const int32_t timeout = int32_t(A(c, 2));
  if (g_poll_logged.fetch_add(1) < 4) cellforge::logf("skate3_community: socketpoll(nfds %d, timeout %d)", nfds, timeout);
  if (nfds < 0 || nfds > 256) return fail_errno(c, kEINVAL);
  std::vector<WSAPOLLFD> pf(size_t(std::max(nfds, 1)));
  std::vector<int> idx(size_t(std::max(nfds, 1)), -1);  // pf slot -> guest index
  int n = 0, invalid = 0, forced = 0;
  for (int i = 0; i < nfds; ++i) {
    const uint32_t e = fds + 8u * uint32_t(i);
    const int fd = int(read_u32(e));
    const uint16_t events = read_u16(e + 4);
    Sock sk;
    write_u16(e + 6, 0);
    if (fd < 0 || !get_sock(fd, sk)) {
      if (fd >= 0) { write_u16(e + 6, 0x20); ++invalid; }  // POLLNVAL
      continue;
    }
    if (events & 0x0001) {  // reply bytes already read ahead for a rewritten web-service call count as readable
      std::lock_guard<std::mutex> lk(g_http_mu);
      auto it = g_http.find(fd);
      if (it != g_http.end() && !it->second.pending.empty()) { write_u16(e + 6, 0x0001); ++forced; continue; }
    }
    short we = 0;
    if (events & (0x0001 | 0x0040)) we |= POLLRDNORM;
    if (events & 0x0004) we |= POLLWRNORM;
    pf[size_t(n)] = WSAPOLLFD{sk.s, we, 0};
    idx[size_t(n)] = i;
    ++n;
  }
  int ready = invalid + forced;
  if (n > 0) {
    // The wait is split so a long poll doesn't hold a socket that closes underneath it for too long.
    int to_ms = forced ? 0 : (timeout < 0 ? -1 : timeout);
    const int rc = WSAPoll(pf.data(), ULONG(n), to_ms);
    if (rc == SOCKET_ERROR) return fail_wsa(c);
    for (int k = 0; k < n; ++k) {
      const short re = pf[size_t(k)].revents;
      uint16_t out = 0;
      if (re & (POLLRDNORM | POLLIN)) out |= 0x0001;
      if (re & POLLWRNORM) out |= 0x0004;
      if (re & POLLERR) out |= 0x0008;
      if (re & POLLHUP) out |= 0x0010;
      if (re & POLLNVAL) out |= 0x0020;
      if (out) { write_u16(fds + 8u * uint32_t(idx[size_t(k)]) + 6, out); ++ready; }
    }
  } else if (timeout > 0 && ready == 0) {
    Sleep(DWORD(timeout));
  }
  RET(c, ready);
}

// Hostnames: the EA servers are redirected to the community server; everything else resolves normally.
bool g_hent_ready = false;
uint32_t g_hent = 0;
std::mutex g_hent_mu;

// '*' matches any run of characters ("downloads.skate*.online.ea.com").
bool glob_match(const char* pat, const char* str) {
  if (!*pat) return !*str;
  if (*pat == '*') {
    for (const char* s = str;; ++s) {
      if (glob_match(pat + 1, s)) return true;
      if (!*s) return false;
    }
  }
  return *pat == *str && glob_match(pat + 1, str + 1);
}

bool resolve_host(const std::string& name, in_addr& out) {
  const std::string l = lower(name);
  for (const auto& h : g_cfg.redirect_hosts) {
    if (glob_match(h.c_str(), l.c_str())) {
      in_addr a{};
      if (inet_pton(AF_INET, g_cfg.server_ip.c_str(), &a) == 1) {
        out = a;
        return true;
      }
      // server_ip may be a hostname
      addrinfo hints{}, *res = nullptr;
      hints.ai_family = AF_INET;
      if (getaddrinfo(g_cfg.server_ip.c_str(), nullptr, &hints, &res) == 0 && res) {
        out = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
        freeaddrinfo(res);
        return true;
      }
      return false;
    }
  }
  in_addr lit{};
  if (inet_pton(AF_INET, name.c_str(), &lit) == 1) { out = lit; return true; }
  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_INET;
  if (getaddrinfo(name.c_str(), nullptr, &hints, &res) != 0 || !res) return false;
  out = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
  freeaddrinfo(res);
  return true;
}

void h_gethostbyname(Context& c) {
  ensure_wsa();
  const std::string name = cellforge::read_cstr(A(c, 0), 255);
  in_addr a{};
  const bool ok = resolve_host(name, a);
  char ip[32] = {};
  if (ok) inet_ntop(AF_INET, &a, ip, sizeof ip);
  cellforge::logf("skate3_community: gethostbyname(%s) -> %s", name.c_str(), ok ? ip : "not found");
  if (!ok) return RET(c, 0);
  std::lock_guard<std::mutex> lk(g_hent_mu);
  if (!g_hent_ready) { g_hent = cellforge::guest_alloc(512, 16); g_hent_ready = g_hent != 0; }
  if (!g_hent) return RET(c, 0);
  const uint32_t h = g_hent, aliases = h + 32, addr_list = h + 48, addr = h + 64, nm = h + 96;
  std::memset(G(h), 0, 512);
  std::memcpy(G(nm), name.c_str(), std::min<size_t>(name.size(), 255));
  write_u32(h, nm);          // h_name
  write_u32(h + 4, aliases); // h_aliases (empty list)
  write_u32(h + 8, 2);       // h_addrtype = AF_INET
  write_u32(h + 12, 4);      // h_length
  write_u32(h + 16, addr_list);
  write_u32(addr_list, addr);
  write_u32(addr_list + 4, 0);
  write_u32(addr, ntohl(a.s_addr));  // network order == big-endian guest order
  RET(c, h);
}

void h_inet_addr(Context& c) {
  const std::string s = cellforge::read_cstr(A(c, 0), 63);
  in_addr a{};
  if (inet_pton(AF_INET, s.c_str(), &a) != 1) return RET(c, 0xFFFFFFFFll);  // INADDR_NONE
  RET(c, ntohl(a.s_addr));
}

uint32_t g_ntoa_buf = 0;
void h_inet_ntoa(Context& c) {  // in_addr passed by value
  if (!g_ntoa_buf) g_ntoa_buf = cellforge::guest_alloc(32, 16);
  in_addr a{};
  a.s_addr = htonl(A(c, 0));
  char b[32] = {};
  inet_ntop(AF_INET, &a, b, sizeof b);
  std::memcpy(G(g_ntoa_buf), b, sizeof b);
  RET(c, g_ntoa_buf);
}

// libnet functions whose purpose isn't known yet: log the call once so the arguments can be studied.
std::atomic<int> g_unknown_logged{0};
void h_unknown_net(Context& c) {
  if (g_unknown_logged.fetch_add(1) < 20)
    cellforge::logf("skate3_community: unidentified libnet call args 0x%llX 0x%llX 0x%llX", (unsigned long long)c.r[3],
                    (unsigned long long)c.r[4], (unsigned long long)c.r[5]);
  RET(c, 0);
}

// ---- cellNetCtl --------------------------------------------------------------------------------------------------
uint32_t local_ipv4() {  // host byte order
  ensure_wsa();
  uint32_t ip = 0x7F000001;
  SOCKET s = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (s != INVALID_SOCKET) {
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(53);
    inet_pton(AF_INET, "1.1.1.1", &to.sin_addr);
    if (::connect(s, reinterpret_cast<sockaddr*>(&to), sizeof to) == 0) {
      sockaddr_in me{};
      int l = sizeof me;
      if (getsockname(s, reinterpret_cast<sockaddr*>(&me), &l) == 0) ip = ntohl(me.sin_addr.s_addr);
    }
    closesocket(s);
  }
  return ip;
}

std::string ip_to_string(uint32_t host_order) {
  in_addr a{};
  a.s_addr = htonl(host_order);
  char b[32];
  inet_ntop(AF_INET, &a, b, sizeof b);
  return b;
}

void put_string(uint32_t ea, const std::string& s, size_t cap) {
  std::memset(G(ea), 0, cap);
  std::memcpy(G(ea), s.c_str(), std::min(s.size(), cap - 1));
}

void h_netctl_get_state(Context& c) { write_u32(A(c, 0), 3); RET(c, 0); }  // CELL_NET_CTL_STATE_IPObtained

void h_netctl_get_info(Context& c) {  // (code, CellNetCtlInfo*)
  const uint32_t code = A(c, 0), info = A(c, 1);
  if (!info) return RET(c, 0x80130104);  // CELL_NET_CTL_ERROR_INVALID_ADDR
  const uint32_t ip = local_ipv4();
  switch (code) {
    case 1: write_u32(info, 0); break;                                   // DEVICE: wired
    case 2: std::memset(G(info), 0, 8); write_u8(info, 0x02); write_u8(info + 5, 0x01); break;  // ETHER_ADDR
    case 3: write_u32(info, 1500); break;                                // MTU
    case 4: write_u32(info, 1); break;                                   // LINK: connected
    case 5: write_u32(info, 0); break;                                   // LINK_TYPE
    case 13: write_u32(info, 0); break;                                  // IP_CONFIG: DHCP
    case 16: put_string(info, ip_to_string(ip), 16); break;              // IP_ADDRESS
    case 17: put_string(info, "255.255.255.0", 16); break;               // NETMASK
    case 18: put_string(info, ip_to_string((ip & 0xFFFFFF00u) | 1u), 16); break;  // DEFAULT_ROUTE
    case 19: put_string(info, "1.1.1.1", 16); break;                     // PRIMARY_DNS
    case 20: put_string(info, "8.8.8.8", 16); break;                     // SECONDARY_DNS
    case 21: write_u32(info, 0); break;                                  // HTTP_PROXY_CONFIG: off
    case 24: write_u32(info, 0); break;                                  // UPNP_CONFIG
    default: std::memset(G(info), 0, 64); break;
  }
  RET(c, 0);
}

void h_netctl_get_nat_info(Context& c) {  // CellNetCtlNatInfo { size, upnp_status, stun_status, nat_type, mapped_addr }
  const uint32_t p = A(c, 0);
  if (!p) return RET(c, 0x80130104);
  write_u32(p + 4, 1);   // UPNP: not used
  write_u32(p + 8, 2);   // STUN: ok
  write_u32(p + 12, 2);  // NAT type 2
  write_u32(p + 16, local_ipv4());
  RET(c, 0);
}

void h_netctl_dialog_unload(Context& c) {  // (CellNetCtlNetStartDialogResult* { u32 size; s32 result })
  if (const uint32_t r = A(c, 0)) write_u32(r + 4, 0);
  RET(c, 0);
}

// ---- sceNp -----------------------------------------------------------------------------------------------------------
// SCE_NP_MANAGER_STATUS_*: OFFLINE -1, GETTING_TICKET 0, GETTING_PROFILE 1, LOGGING_IN 2, ONLINE 3. Event 255 = GOT_TICKET.
constexpr int kNpOffline = -1, kNpLoggingIn = 2, kNpOnline = 3;
constexpr uint32_t kNpErrorOffline = 0x8002aa0c;

struct NpState {
  std::mutex mu;
  uint32_t cb = 0, cb_arg = 0;        // sceNpManagerRegisterCallback
  bool status_dirty = false;          // owe the callback a status event
  bool ticket_event = false;          // owe the callback a GOT_TICKET event
  int ticket_result = 0;              // its result: ticket size, or a negative error
  uint32_t netctl_handler = 0, netctl_arg = 0;
  bool deliver_netctl = false;
  std::string online_name;            // the signed-in RPCN name (shown to the game as the NP id)
  std::vector<uint8_t> ticket;        // latest signed ticket
  std::string ticket_service;
  std::chrono::steady_clock::time_point ticket_time{};
} g_np;
std::atomic<int> g_np_status{kNpLoggingIn};
std::atomic<bool> g_signin_busy{false};
std::atomic<bool> g_signin_declined{false};  // the player chose "Not now": stay offline for this session

void np_set_status(int s) {
  g_np_status = s;
  std::lock_guard<std::mutex> lk(g_np.mu);
  g_np.status_dirty = true;
}
void np_finish_ticket(int result) {
  std::lock_guard<std::mutex> lk(g_np.mu);
  g_np.ticket_event = true;
  g_np.ticket_result = result;
}
std::string np_name() {
  std::lock_guard<std::mutex> lk(g_np.mu);
  return g_np.online_name.empty() ? g_cfg.np_name : g_np.online_name;
}

// Makes sure there is a saved RPCN login, asking the player (once) when there isn't. A login entered in the window is
// checked with RPCN right away (so a wrong password or a missing email token is fixed in the same window) and saved.
// Returns false if the player declined.
bool ensure_credentials(s3c::Credentials& cred, s3c::SignInForm& form) {
  if (s3c::load_credentials(cred)) return true;
  if (g_signin_declined) return false;
  if (!g_cfg.prompt_signin) return false;
  for (;;) {
    switch (s3c::prompt_signin(form)) {
      case s3c::SignInAction::SignIn: {
        if (form.npid.empty() || form.password.empty()) { form.message = "Please enter your RPCN username and password."; continue; }
        const auto r = s3c::rpcn_login_and_ticket(g_cfg.rpcn_host, uint16_t(g_cfg.rpcn_port), form.npid, form.password, form.token, "", {});
        if (r.ok) {
          cred = {form.npid, form.password, form.token};
          s3c::save_credentials(cred);
          return true;
        }
        cellforge::logf("skate3_community: RPCN sign-in: %s", r.message.c_str());
        form.message = r.error_code == 9 ? "RPCN needs the verification token from the email it sent you. Enter it below."
                                          : "RPCN says: " + r.message + ". Please check your details and try again.";
        if (r.credentials_bad) form.password.clear();
        continue;
      }
      case s3c::SignInAction::CreateAccount: {
        form.message.clear();
        while (s3c::prompt_create_account(form)) {
          const auto r = s3c::rpcn_create_account(g_cfg.rpcn_host, uint16_t(g_cfg.rpcn_port), form.npid, form.password, form.email);
          if (r.ok) {
            form.message = r.message + ". Enter the token from the email below, then press Sign in.";
            break;
          }
          form.message = "Could not create the account: " + r.message;
        }
        continue;
      }
      case s3c::SignInAction::NotNow:
        g_signin_declined = true;
        return false;
    }
  }
}

// Initial sign-in when the game starts using the NP library: get the saved (or newly entered) login.
void signin_thread() {
  if (g_cfg.dev_fake_signin) {
    {
      std::lock_guard<std::mutex> lk(g_np.mu);
      g_np.online_name = "DevUser";
    }
    cellforge::log("skate3_community: dev_fake_signin: pretending to be signed in");
    np_set_status(kNpOnline);
    g_signin_busy = false;
    return;
  }
  s3c::Credentials cred;
  s3c::SignInForm form;
  if (ensure_credentials(cred, form)) {
    {
      std::lock_guard<std::mutex> lk(g_np.mu);
      g_np.online_name = cred.npid;
    }
    cellforge::logf("skate3_community: signed in as %s (ticket requested when the game needs it)", cred.npid.c_str());
    np_set_status(kNpOnline);
  } else {
    cellforge::log("skate3_community: no RPCN login; staying offline");
    np_set_status(kNpOffline);
  }
  g_signin_busy = false;
}

// Gets a fresh ticket from RPCN for the game's service id; on a bad login asks the player again.
void ticket_thread(std::string service_id, std::vector<uint8_t> cookie) {
  if (g_cfg.dev_fake_signin) {
    {
      std::lock_guard<std::mutex> lk(g_np.mu);
      g_np.ticket.assign(256, 0x5A);
      g_np.ticket_service = service_id;
      g_np.ticket_time = std::chrono::steady_clock::now();
    }
    np_set_status(kNpOnline);
    np_finish_ticket(256);
    g_signin_busy = false;
    return;
  }
  s3c::Credentials cred;
  s3c::SignInForm form;
  int attempts = 0;
  for (;;) {
    if (!ensure_credentials(cred, form)) { np_set_status(kNpOffline); np_finish_ticket(int(kNpErrorOffline)); break; }
    const auto r = s3c::rpcn_login_and_ticket(g_cfg.rpcn_host, uint16_t(g_cfg.rpcn_port), cred.npid, cred.password, cred.token, service_id, cookie);
    if (r.ok) {
      s3c::save_credentials(cred);
      {
        std::lock_guard<std::mutex> lk(g_np.mu);
        g_np.online_name = cred.npid;
        g_np.ticket = r.ticket;
        g_np.ticket_service = service_id;
        g_np.ticket_time = std::chrono::steady_clock::now();
      }
      cellforge::logf("skate3_community: got an RPCN ticket (%zu bytes) for %s", r.ticket.size(), service_id.c_str());
      np_set_status(kNpOnline);
      np_finish_ticket(int(r.ticket.size()));
      break;
    }
    cellforge::logf("skate3_community: RPCN: %s", r.message.c_str());
    if (r.credentials_bad) {  // wrong name / password / token: ask again
      s3c::forget_credentials();
      form = {cred.npid, "", "", "", "RPCN says: " + r.message + ". Please sign in again."};
      cred = {};
      if (r.error_code == 9) form.message = "RPCN needs the verification token from the email it sent you. Enter it below.";
      // Keep the token the player types next in this session.
      s3c::SignInAction act = s3c::prompt_signin(form);
      if (act == s3c::SignInAction::SignIn) { cred = {form.npid, form.password, form.token}; continue; }
      g_signin_declined = true;
      np_set_status(kNpOffline);
      np_finish_ticket(int(kNpErrorOffline));
      break;
    }
    if (r.error_code == 6 && ++attempts < 4) { Sleep(2500); continue; }  // previous session still closing
    np_set_status(kNpOffline);
    np_finish_ticket(int(kNpErrorOffline));
    break;
  }
  g_signin_busy = false;
}

void start_worker(void (*fn)()) {
  if (g_signin_busy.exchange(true)) return;
  std::thread(fn).detach();
}

void h_np_get_status(Context& c) { cellforge::write_i32(A(c, 0), g_np_status.load()); RET(c, 0); }

void h_np_get_npid(Context& c) {  // SceNpId: handle[16] + dummy[4], opt[8], reserved[8]
  const uint32_t p = A(c, 0);
  if (!p) return RET(c, 0x8002aa04);  // SCE_NP_ERROR_INVALID_ARGUMENT
  if (g_np_status.load() == kNpOffline) return RET(c, kNpErrorOffline);
  const std::string name = np_name();
  std::memset(G(p), 0, 36);
  std::memcpy(G(p), name.c_str(), std::min<size_t>(name.size(), 16));
  RET(c, 0);
}

void h_np_register_callback(Context& c) {  // (callback, arg)
  {
    std::lock_guard<std::mutex> lk(g_np.mu);
    g_np.cb = A(c, 0);
    g_np.cb_arg = A(c, 1);
    g_np.status_dirty = true;
  }
  cellforge::log("skate3_community: sceNpManagerRegisterCallback");
  static std::atomic<bool> started{false};
  if (!started.exchange(true)) start_worker(signin_thread);
  RET(c, 0);
}

void h_np_get_ticket(Context& c) {  // (void* buffer, u32* length)
  const uint32_t buf = A(c, 0), lenp = A(c, 1);
  if (!lenp) return RET(c, 0x8002aa04);
  std::lock_guard<std::mutex> lk(g_np.mu);
  const uint32_t have = uint32_t(g_np.ticket.size());
  if (!have) return RET(c, kNpErrorOffline);
  if (!buf) { write_u32(lenp, have); return RET(c, 0); }
  const uint32_t n = std::min(have, read_u32(lenp));
  std::memcpy(G(buf), g_np.ticket.data(), n);
  write_u32(lenp, n);
  RET(c, 0);
}

void h_np_request_ticket2(Context& c) {  // (npid*, version*, serviceId*, cookie*, cookieSize, entitlementId, consumedCount)
  const std::string service = A(c, 2) ? cellforge::read_cstr(A(c, 2), 24) : std::string();
  std::vector<uint8_t> cookie;
  if (A(c, 3) && A(c, 4) && A(c, 4) <= 1024) {
    cookie.resize(A(c, 4));
    cellforge::read_bytes(A(c, 3), cookie.data(), cookie.size());
  }
  cellforge::logf("skate3_community: sceNpManagerRequestTicket2(%s, cookie %zu bytes)", service.c_str(), cookie.size());
  {
    // A ticket for the same service that is still fresh (they last 15 minutes) is reused.
    std::lock_guard<std::mutex> lk(g_np.mu);
    if (!g_np.ticket.empty() && g_np.ticket_service == service &&
        std::chrono::steady_clock::now() - g_np.ticket_time < std::chrono::minutes(8)) {
      g_np.ticket_event = true;
      g_np.ticket_result = int(g_np.ticket.size());
      return RET(c, 0);
    }
  }
  if (g_np_status.load() == kNpOffline && g_signin_declined) {
    np_finish_ticket(int(kNpErrorOffline));
    return RET(c, 0);
  }
  static std::string s_service;
  static std::vector<uint8_t> s_cookie;
  // Worker threads are started with their arguments captured by value.
  if (!g_signin_busy.exchange(true)) {
    std::thread([service, cookie] { ticket_thread(service, cookie); }).detach();
  } else {
    // A sign-in prompt is still open: answer once it is done by retrying shortly.
    std::thread([service, cookie] {
      while (g_signin_busy.load()) Sleep(200);
      if (!g_signin_busy.exchange(true)) ticket_thread(service, cookie);
    }).detach();
  }
  (void)s_service; (void)s_cookie;
  RET(c, 0);
}

void h_netctl_add_handler(Context& c) {  // (handler, arg, s32* id)
  {
    std::lock_guard<std::mutex> lk(g_np.mu);
    g_np.netctl_handler = A(c, 0);
    g_np.netctl_arg = A(c, 1);
    g_np.deliver_netctl = true;
  }
  if (A(c, 2)) write_u32(A(c, 2), 1);
  RET(c, 0);
}

// Calls a guest function pointer (an OPD) from inside a hook, with up to five integer arguments.
void call_guest(const Context& base, uint32_t opd, uint64_t a0, uint64_t a1 = 0, uint64_t a2 = 0, uint64_t a3 = 0, uint64_t a4 = 0) {
  Context nc = base;
  nc.r[2] = read_u32(opd + 4);
  nc.r[3] = a0;
  nc.r[4] = a1;
  nc.r[5] = a2;
  nc.r[6] = a3;
  nc.r[7] = a4;
  cellforge::call(nc, read_u32(opd));
}

// ---- the game's embedded-browser front-end state (Skate 3 only) ----
// The front end's "webkit" state shows web pages (the news feed, the online hub, ...) and ends when the page's script sends it
// the command "quitwebkit". CellForge has no browser, so no page ever loads and the state would wait forever. When the state is
// entered, a little later this mod pushes that command into the browser manager's event queue itself (the same game function the
// real browser thread uses), which sends the front end on as if the page had closed.
std::atomic<int> g_webkit_quit_countdown{0};
std::atomic<uint32_t> g_webkit_mgr{0};
// The browser state was entered. Start-up and later entries alike: the community server's pages only show a credits box and leave.
void on_webkit_state_entered() {
  if (!g_webkit_skip_prompt) g_webkit_quit_countdown = g_webkit_credits ? 360 : 90;
}
constexpr uint32_t kSkate3WebkitPushEvent = 0x004C8F64;   // Manager::PushEvent(this, const wstring*)
void push_webkit_command(const Context& c, const char* text) {
  const uint32_t mgr = g_webkit_mgr.load();
  if (!mgr) return;
  const uint32_t len = uint32_t(std::strlen(text));
  const uint32_t str = cellforge::guest_alloc(16 + 2 * (len + 1), 16);
  if (!str) return;
  write_u32(str, str + 16);              // begin
  write_u32(str + 4, str + 16 + 2 * len); // end
  write_u32(str + 8, str + 16 + 2 * (len + 1));
  for (uint32_t i = 0; i < len; ++i) { write_u8(str + 16 + 2 * i, 0); write_u8(str + 16 + 2 * i + 1, uint8_t(text[i])); }
  Context nc = c;
  nc.r[3] = mgr;
  nc.r[4] = str;
  cellforge::call(nc, kSkate3WebkitPushEvent);
  cellforge::logf("skate3_community: told the front end the web page sent \"%s\"", text);
}

void deliver_commerce_events(const Context& c);
// sceNp functions this mod doesn't implement yet: log how the game calls them and answer "OK" (the same as the runtime default).
template <uint32_t NID>
void h_np_unknown(Context& c) {
  static std::atomic<int> n{0};
  if (n.fetch_add(1) < 12)
    cellforge::logf("skate3_community: unimplemented sceNp 0x%08X: r3=0x%llX r4=0x%llX r5=0x%llX r6=0x%llX", NID,
                    (unsigned long long)c.r[3], (unsigned long long)c.r[4], (unsigned long long)c.r[5], (unsigned long long)c.r[6]);
  c.r[3] = 0;
}
ppu::Fn g_check_orig = nullptr;
void h_sysutil_check_callback(Context& c) {
  if (g_check_orig) g_check_orig(c);
  const uint64_t result = c.r[3];
  uint32_t cb = 0, arg = 0, nh = 0, narg = 0;
  bool status = false, ticket = false, netctl = false;
  int ticket_result = 0;
  {
    std::lock_guard<std::mutex> lk(g_np.mu);
    cb = g_np.cb; arg = g_np.cb_arg; nh = g_np.netctl_handler; narg = g_np.netctl_arg;
    status = g_np.status_dirty && cb;
    ticket = g_np.ticket_event && cb;
    ticket_result = g_np.ticket_result;
    netctl = g_np.deliver_netctl && nh;
    if (status) g_np.status_dirty = false;
    if (ticket) g_np.ticket_event = false;
    if (netctl) g_np.deliver_netctl = false;
  }
  // cellNetCtlHandler(prev_state, new_state, event, error_code, arg): disconnected -> IP obtained (GET_IP).
  if (netctl) call_guest(c, nh, 0, 3, 2, 0, narg);
  // SceNpManagerCallback(event, result, arg): the new status, then GOT_TICKET (255) with the ticket size.
  if (status) call_guest(c, cb, uint64_t(int64_t(g_np_status.load())), 0, arg);
  if (ticket) call_guest(c, cb, 255, uint64_t(int64_t(ticket_result)), arg);
  deliver_commerce_events(c);
  if (g_webkit_quit_countdown.load() > 0 && g_webkit_quit_countdown.fetch_sub(1) == 1) push_webkit_command(c, "quitwebkit");
  c.r[3] = result;
}

// ---- sceNpCommerce2 (the PlayStation Store library; Skate 3 uses it for its "scanning for downloadable content" step) ----
// There is no store behind this: sessions "open" at once and every catalogue query reports that the store has nothing, which
// is what lets the game finish its scan and continue (the same behaviour RPCS3 gives the game).
struct Commerce {
  struct Ctx { uint32_t handler = 0, arg = 0; };
  struct Event { uint32_t handler, ctx, event, arg; int32_t error; };
  std::mutex mu;
  std::unordered_map<uint32_t, Ctx> ctxs;
  std::vector<Event> events;
  uint32_t next_ctx = 1;
} g_com;

constexpr uint32_t kComCtxNotFound = 0x8002300c, kComInvalidArg = 0x80023003, kComBadResult = 0x80023014,
                   kComMaintenance = 0x80023016, kComBufTooSmall = 0x80023018;

void h_com_create_ctx(Context& c) {  // (version, SceNpId*, handler, arg, u32* ctx_id)
  if (A(c, 0) > 2) return RET(c, 0x80023004);
  if (!A(c, 1)) return RET(c, kComInvalidArg);
  std::lock_guard<std::mutex> lk(g_com.mu);
  const uint32_t id = g_com.next_ctx++;
  g_com.ctxs[id] = {A(c, 2), A(c, 3)};
  if (A(c, 4)) write_u32(A(c, 4), id);
  tracef("sceNpCommerce2CreateCtx -> %u", id);
  RET(c, 0);
}
void h_com_destroy_ctx(Context& c) {
  std::lock_guard<std::mutex> lk(g_com.mu);
  RET(c, g_com.ctxs.erase(A(c, 0)) ? 0 : kComCtxNotFound);
}
void h_com_session_start(Context& c) {  // (ctx_id) -> event CREATE_SESSION_DONE (0x0011)
  std::lock_guard<std::mutex> lk(g_com.mu);
  auto it = g_com.ctxs.find(A(c, 0));
  if (it == g_com.ctxs.end()) return RET(c, kComCtxNotFound);
  if (it->second.handler) g_com.events.push_back({it->second.handler, A(c, 0), 0x0011, it->second.arg, 0});
  RET(c, 0);
}
void h_com_session_finish(Context& c) {  // (ctx_id, SceNpCommerce2SessionInfo*)
  {
    std::lock_guard<std::mutex> lk(g_com.mu);
    if (!g_com.ctxs.count(A(c, 0))) return RET(c, kComCtxNotFound);
  }
  if (A(c, 1)) std::memset(G(A(c, 1)), 0, 44);
  RET(c, 0);
}
void h_com_ok(Context& c) { RET(c, 0); }
void h_com_category_info(Context& c) { RET(c, kComMaintenance); }       // no categories
void h_com_bad_result(Context& c) { RET(c, kComBadResult); }            // Init*Result: no result data
void h_com_get_result(Context& c) { RET(c, A(c, 2) < 262144 ? kComBufTooSmall : 0); }  // (req, buf, buf_size, fill_size*)

void h_np_friend_count(Context& c) {  // sceNpBasicGetFriendListEntryCount(u32* count): no friends
  if (!A(c, 0)) return RET(c, 0x8002a004);
  write_u32(A(c, 0), 0);
  RET(c, 0);
}

void deliver_commerce_events(const Context& c) {
  std::vector<Commerce::Event> evs;
  {
    std::lock_guard<std::mutex> lk(g_com.mu);
    evs.swap(g_com.events);
  }
  // SceNpCommerce2Handler(ctx_id, subject_id, event, error_code, arg)
  for (const auto& e : evs) call_guest(c, e.handler, e.ctx, 0, e.event, uint64_t(int64_t(e.error)), e.arg);
}

// ---- EA WebKit stand-in ---------------------------------------------------------------------------------------------------
// Skate 3 ships its in-game browser (EAWebKit) as a separate game module that CellForge cannot load. When the game goes online
// it asks that module for its interface object and uses it without checking for null. We hand back an object whose every
// virtual function does nothing and returns 0 ("no browser available").
uint32_t g_webkit_obj = 0;
// Every virtual function returns the stand-in object itself (so "give me a sub-object" calls get something valid to call
// into) and logs its first calls, which shows what the game wants from the browser.
template <int N>
void h_webkit_slot(Context& c) {
  static std::atomic<int> calls{0};
  if (calls.fetch_add(1) < 3)
    cellforge::logf("skate3_community: EAWebKit virtual #%d called (r3=0x%llX r4=0x%llX r5=0x%llX)", N, (unsigned long long)c.r[3],
                    (unsigned long long)c.r[4], (unsigned long long)c.r[5]);
  c.r[3] = g_webkit_obj;
}
constexpr int kWebkitSlots = 64;
template <size_t... I>
std::array<ppu::Fn, sizeof...(I)> make_webkit_slots(std::index_sequence<I...>) {
  return {h_webkit_slot<int(I)>...};
}
void h_webkit_get_interface(Context& c) {
  if (!g_webkit_obj) {
    const uint32_t obj = cellforge::guest_alloc(1024, 16);
    const uint32_t table = cellforge::guest_alloc(4 * 512, 16);
    if (!obj || !table) return RET(c, 0);
    g_webkit_obj = obj;
    static const auto slots = make_webkit_slots(std::make_index_sequence<kWebkitSlots>{});
    uint32_t last = 0;
    for (uint32_t i = 0; i < 512; ++i) {
      if (i < kWebkitSlots) last = cellforge::make_opd(slots[i]);
      write_u32(table + 4 * i, last);
    }
    write_u32(obj, table);
    cellforge::logf("skate3_community: EAWebKit interface requested; providing an inert stand-in at 0x%08X", obj);
  }
  RET(c, g_webkit_obj);
}

// Skate 3 (BLUS30464) specific: the game's own "start the embedded browser" routine. It fetches the EAWebKit interface from the
// bundled module, stores it in a global and starts a 'webkit' thread. Skipping it keeps the game from ever needing a browser
// (the game treats a browser that was never started as "not available").
constexpr uint32_t kSkate3WebkitInit = 0x004C7BE4;
constexpr uint32_t kSkate3WebkitInitFirstWord = 0xF821FF71;  // stdu r1,-144(r1)
// The game's 'webkit' thread entry. It calls the routine above and then uses the browser object without checking it, so the
// thread must not run either: returning immediately ends the thread.
constexpr uint32_t kSkate3WebkitThread = 0x004C9C84;
constexpr uint32_t kSkate3WebkitThreadFirstWord = 0xF821FF21;  // stdu r1,-224(r1)
void h_skip_webkit_thread(Context& c) {
  cellforge::log("skate3_community: ending the game's 'webkit' thread (no embedded browser in CellForge)");
  // The thread normally runs the start-up routine first, which ends by setting the manager's "initialised" byte; the front
  // end waits for it, so set it (r3 is the manager object).
  const uint32_t self = uint32_t(c.r[3]);
  if (self >= 0x10000) write_u8(self + 1107, 1);
  // The front end also checks the manager's "browser ready" byte (this+1104) before it moves on from its browser state.
  if (self >= 0x10000) write_u8(self + 1104, 1);
  if (self >= 0x10000) g_webkit_mgr = self;
  c.r[3] = 0;
}
void h_skip_webkit_init(Context& c) {
  cellforge::log("skate3_community: skipping the game's embedded browser start-up (not available in CellForge)");
  // The real routine ends by setting the manager's "initialised" byte (this+1107); the front end waits for it, so set it.
  const uint32_t self = uint32_t(c.r[3]);
  if (self >= 0x10000) write_u8(self + 1107, 1);
  c.r[3] = 0;
}

// Skate 3 specific troubleshooting aid: logs calls to a few of the game's own sign-in state-machine functions (wrapping them,
// not replacing them) so the order of events around the online sign-in is visible in the log.
template <uint32_t ADDR>
struct FlowWrap {
  static inline ppu::Fn orig = nullptr;
  static inline std::atomic<int> count{0};
  static void fn(Context& c) {
    const uint32_t r3 = uint32_t(c.r[3]), r4 = uint32_t(c.r[4]), r5 = uint32_t(c.r[5]);
    const int n = count.fetch_add(1);
    static std::atomic<uint32_t> signin_obj{0};
    if (ADDR == 0x0043C454) signin_obj = r3;
    if (ADDR == 0x0043C454 && r4 == 1) {
      static std::atomic<int> ticks{0};
      const int t = ticks.fetch_add(1) + 1;
      if (t % 120 == 0) cellforge::logf("skate3_community: [flow] sign-in controller ticks so far: %d", t);
    }
    bool show = n < 300;
    if (ADDR == 0x0043C454 && r4 == 1 && n > 3) show = false;          // per-frame tick
    if (ADDR == 0x00418DF4 && r4 == 1) show = false;
    if (ADDR == 0x0040BA04) show = n < 1500;
    if (show) {
      if (ADDR == 0x0040BA04) {
        // EA state machine: ChangeState(fsm = r3, new state = member-function pointer packed in r4 {ptr (high), adjust (low)}).
        const uint32_t hi = uint32_t(c.r[4] >> 32), lo = uint32_t(c.r[4]);
        uint32_t code = 0;
        if (!(hi & 1) && hi >= 0x10000 && hi < 0x02000000) code = read_u32(hi);
        cellforge::logf("skate3_community: [fsm] 0x%08X -> state %s0x%08X (code 0x%08X, adj 0x%X) r5=%u", r3, (hi & 1) ? "virtual slot " : "", (hi & 1) ? hi - 1 : hi, code, lo, r5);
      } else if (ADDR == 0x0043C454)
        cellforge::logf("skate3_community: [flow] SignIn::OnEvent(this=0x%08X, event=%u) flags 712/713/714 = %u/%u/%u", r3, r4,
                        read_u8(r3 + 712), read_u8(r3 + 713), read_u8(r3 + 714));
      else
        cellforge::logf("skate3_community: [flow] fn 0x%08X(r3=0x%08X r4=0x%08X r5=0x%08X)", ADDR, r3, r4, r5);
    }
    if (ADDR == 0x0040BA04 && uint32_t(c.r[4] >> 32) == 0x017BB760) on_webkit_state_entered();  // the browser state was entered
    // With no browser the game would ask "are you sure you want to exit the web browser?" (a Yes/No prompt over nothing) before
    // leaving the page. Go straight to the state that follows the page closing instead.
    if (ADDR == 0x0040BA04 && g_webkit_skip_prompt && uint32_t(c.r[4] >> 32) == 0x017BA360) {
      c.r[4] = (uint64_t(0x017BAD88) << 32) | uint32_t(c.r[4]);
      cellforge::log("skate3_community: skipped the browser exit prompt");
    }
    orig(c);
    if (show && ADDR == 0x0043C454)
      cellforge::logf("skate3_community: [flow]   -> flags 712/713/714 = %u/%u/%u", read_u8(r3 + 712), read_u8(r3 + 713), read_u8(r3 + 714));
  }
};
template <uint32_t ADDR>
void flow_hook(uint32_t first_word) {
  if (read_u32(ADDR) != first_word) return;
  FlowWrap<ADDR>::orig = cellforge::hook(ADDR, &FlowWrap<ADDR>::fn);
}

// Troubleshooting (Skate 3): log the text ids the game puts into its pop-up dialogs (message / title / button labels).
std::string peek_cstr(uint32_t p) {
  std::string out;
  if (p < 0x10000 || p >= 0xD0000000u) return out;
  for (int i = 0; i < 80; ++i) {
    const uint8_t b = read_u8(p + uint32_t(i));
    if (b == 0) break;
    out += (b >= 32 && b < 127) ? char(b) : '?';
  }
  return out;
}
template <uint32_t ADDR, int STRARG, const char* WHAT>
struct DialogLog {
  static inline ppu::Fn orig = nullptr;
  static void fn(Context& c) {
    const uint32_t sp = uint32_t(c.r[STRARG]);
    const std::string t = peek_cstr(sp);
    if (!t.empty()) cellforge::logf("skate3_community: [dialog] %s: %s", WHAT, t.c_str());
    if (orig) orig(c);
  }
};
constexpr char kWhatMsg[] = "message", kWhatTitle[] = "title", kWhatBtn[] = "button";
template <uint32_t ADDR, int STRARG, const char* WHAT>
void dialog_log_hook(uint32_t first_word) {
  if (read_u32(ADDR) != first_word) return;
  DialogLog<ADDR, STRARG, WHAT>::orig = cellforge::hook(ADDR, &DialogLog<ADDR, STRARG, WHAT>::fn);
}

template <uint32_t ADDR>
struct RetLog {
  static inline ppu::Fn orig = nullptr;
  static inline std::atomic<int> n{0};
  static void fn(Context& c) {
    const uint32_t r3 = uint32_t(c.r[3]);
    const uint8_t b2612 = ADDR == 0x0040FC8C ? read_u8(r3 + 2612) : 0, b2614 = ADDR == 0x0040FC8C ? read_u8(r3 + 2614) : 0;
    if (orig) orig(c);
    if (n.fetch_add(1) < 6)
      cellforge::logf("skate3_community: [ret] fn 0x%08X(r3=0x%08X) -> %lld (state obj +2612=%u +2614=%u)", ADDR, r3,
                      (long long)int64_t(c.r[3]), unsigned(b2612), unsigned(b2614));
  }
};
template <uint32_t ADDR>
void ret_log_hook(uint32_t first_word) {
  if (read_u32(ADDR) != first_word) return;
  RetLog<ADDR>::orig = cellforge::hook(ADDR, &RetLog<ADDR>::fn);
}

// The community server's web pages (every "/skate3/webkit/..." address) only show this text in a message box and then leave. The
// game's exit prompt is used as the box: its text is swapped for the credits (the game's text system shows plain text as it is).
constexpr const char* kCreditsTitle = "Skate 3 Community Server";
constexpr const char* kCreditsText =
    "Skate 3 Custom Blaze Server by Wispp!^^Special thanks for prior Blaze/TDF reversing efforts:^^Aim4kill^gamingrobot^Tratos";
uint32_t guest_cstr(const char* t) {
  static std::map<const char*, uint32_t> cache;
  auto it = cache.find(t);
  if (it != cache.end()) return it->second;
  const uint32_t len = uint32_t(std::strlen(t));
  const uint32_t g = cellforge::guest_alloc(len + 1, 16);
  if (!g) return 0;
  for (uint32_t i = 0; i <= len; ++i) write_u8(g + i, uint8_t(t[i]));
  cache[t] = g;
  return g;
}
bool g_credits_dialog = false;  // set while the browser-exit prompt is being built
template <uint32_t ADDR, bool TITLE>
struct CreditsText {
  static inline ppu::Fn orig = nullptr;
  static void fn(Context& c) {
    const std::string t = peek_cstr(uint32_t(c.r[4]));
    if (!TITLE && t == "ID_WEBKIT_EXIT_CONFIRMATION_MESSAGE") {
      g_credits_dialog = true;
      if (const uint32_t g = guest_cstr(kCreditsText)) c.r[4] = g;
      cellforge::log("skate3_community: [credits] put the community server credits text into the browser-exit dialog");
    } else if (TITLE && g_credits_dialog && t == "ID_COMMON_WARNING_TITLE") {
      if (const uint32_t g = guest_cstr(kCreditsTitle)) c.r[4] = g;
    }
    if (orig) orig(c);
  }
};
template <uint32_t ADDR, bool TITLE>
void credits_hook(uint32_t first_word) {
  if (read_u32(ADDR) != first_word) return;
  CreditsText<ADDR, TITLE>::orig = cellforge::hook(ADDR, &CreditsText<ADDR, TITLE>::fn);
}

// ---- installation ------------------------------------------------------------------------------------------------------
int g_hooked = 0, g_missing = 0;
void install(const char* module, const char* name, ppu::Fn fn) {
  if (cellforge::hook_import(module, name, fn)) ++g_hooked;
  else {
    ++g_missing;
    cellforge::logf("skate3_community: %s::%s is not imported by this game (skipped)", module, name);
  }
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;
  load_settings();
  if (!g_cfg.enabled) {
    cellforge::log("skate3_community: disabled in the ini (enabled=0)");
    return 0;
  }
  g_errno_addr = cellforge::guest_alloc(16, 16);
  if (!g_errno_addr) return 2;

  // sys_net
  install("sys_net", "sys_net_initialize_network_ex", h_initialize_network_ex);
  install("sys_net", "sys_net_finalize_network", h_finalize_network);
  install("sys_net", "_sys_net_errno_loc", h_errno_loc);
  install("sys_net", "sys_net_free_thread_context", h_ok0);
  install("sys_net", "socket", h_socket);
  install("sys_net", "socketclose", h_socketclose);
  install("sys_net", "shutdown", h_shutdown);
  install("sys_net", "connect", h_connect);
  install("sys_net", "bind", h_bind);
  install("sys_net", "listen", h_listen);
  install("sys_net", "accept", h_accept);
  install("sys_net", "getsockname", h_getsockname);
  install("sys_net", "getpeername", h_getpeername);
  install("sys_net", "send", h_send);
  install("sys_net", "recv", h_recv);
  install("sys_net", "sendto", h_sendto);
  install("sys_net", "recvfrom", h_recvfrom);
  install("sys_net", "setsockopt", h_setsockopt);
  install("sys_net", "getsockopt", h_getsockopt);
  install("sys_net", "socketpoll", h_socketpoll);
  install("sys_net", "gethostbyname", h_gethostbyname);
  install("sys_net", "inet_addr", h_inet_addr);
  install("sys_net", "inet_ntoa", h_inet_ntoa);
  for (const char* nid : {"nid:0x27FB339D", "nid:0x8CCF05ED", "nid:0x8D1B77FB"}) install("sys_net", nid, h_unknown_net);

  // cellNetCtl
  install("cellNetCtl", "cellNetCtlGetState", h_netctl_get_state);
  install("cellNetCtl", "cellNetCtlGetInfo", h_netctl_get_info);
  install("cellNetCtl", "cellNetCtlGetNatInfo", h_netctl_get_nat_info);
  install("cellNetCtl", "cellNetCtlAddHandler", h_netctl_add_handler);
  install("cellNetCtl", "cellNetCtlNetStartDialogUnloadAsync", h_netctl_dialog_unload);

  // sceNp
  install("sceNp", "nid:0x4210C462", h_np_unknown<0x4210C462>);
  install("sceNp", "nid:0x58FA4FCD", h_np_unknown<0x58FA4FCD>);
  install("sceNp", "nid:0x4026EAC5", h_np_unknown<0x4026EAC5>);
  install("sceNp", "sceNpManagerGetStatus", h_np_get_status);
  install("sceNp", "sceNpManagerGetNpId", h_np_get_npid);
  install("sceNp", "sceNpManagerRegisterCallback", h_np_register_callback);
  install("sceNp", "sceNpManagerGetTicket", h_np_get_ticket);
  install("sceNp", "sceNpManagerRequestTicket2", h_np_request_ticket2);

  if (g_cfg.webkit_mode == "standin") {
    install("cellPrx_EAWebkitPrx", "nid:0xB84A04AC", h_webkit_get_interface);
  } else if (read_u32(kSkate3WebkitInit) == kSkate3WebkitInitFirstWord && cellforge::hook(kSkate3WebkitInit, h_skip_webkit_init)) {
    ++g_hooked;
    if (read_u32(kSkate3WebkitThread) == kSkate3WebkitThreadFirstWord && cellforge::hook(kSkate3WebkitThread, h_skip_webkit_thread)) ++g_hooked;
  } else {
    cellforge::log("skate3_community: could not find the game's browser start-up routine; using the inert stand-in instead");
    install("cellPrx_EAWebkitPrx", "nid:0xB84A04AC", h_webkit_get_interface);
  }

  if (g_webkit_credits && g_cfg.webkit_mode != "standin") {
    credits_hook<0x00C7BBC4, false>(0x38630080);  // dialog message
  }
  if (g_cfg.flow_trace) {
    ret_log_hook<0x0040FC8C>(0xF821FF81);
    ret_log_hook<0x004C29B0>(0x81230000);
    dialog_log_hook<0x00C7BBC4, 4, kWhatMsg>(0x38630080);   // dialog message id
    dialog_log_hook<0x00C7BBFC, 4, kWhatTitle>(0x7C862378); // dialog title id
    dialog_log_hook<0x00C7BB84, 5, kWhatBtn>(0x54843830);   // dialog button label (r4 = index)
  }
  if (g_cfg.webkit_mode != "standin" && g_cfg.webkit_autoquit) flow_hook<0x0040BA04>(0xF821FF61);  // watches for the browser state
  if (g_cfg.flow_trace) {
    flow_hook<0x0043C454>(0x2B840004);  // sign-in controller OnEvent(this, event)
    flow_hook<0x00423A7C>(0xF821FF41);  // reply handler (the HTTP/Blaze result message)
    flow_hook<0x00418DF4>(0xF821FF81);  // state check
    flow_hook<0x00435EC0>(0x3943004C);  // sign-in start (clears the flags)
    flow_hook<0x0049EA50>(0xF821FF81);  // deliver a result message to its listener
    flow_hook<0x00B27BD0>(0xF821FF51);  // result dispatch
  }

  // sceNpCommerce2: answer the game's store scan (see above)
  install("sceNpCommerce2", "sceNpCommerce2CreateCtx", h_com_create_ctx);
  install("sceNpCommerce2", "sceNpCommerce2DestroyCtx", h_com_destroy_ctx);
  install("sceNpCommerce2", "sceNpCommerce2CreateSessionStart", h_com_session_start);
  install("sceNpCommerce2", "sceNpCommerce2CreateSessionFinish", h_com_session_finish);
  install("sceNpCommerce2", "sceNpCommerce2CreateSessionAbort", h_com_ok);
  install("sceNpCommerce2", "sceNpCommerce2GetCategoryInfo", h_com_category_info);
  install("sceNpCommerce2", "sceNpCommerce2InitGetCategoryContentsResult", h_com_bad_result);
  install("sceNpCommerce2", "sceNpCommerce2InitGetProductInfoResult", h_com_bad_result);
  install("sceNpCommerce2", "sceNpCommerce2GetCategoryContentsGetResult", h_com_get_result);
  install("sceNpCommerce2", "sceNpCommerce2GetProductInfoGetResult", h_com_get_result);
  install("sceNp", "sceNpBasicGetFriendListEntryCount", h_np_friend_count);

  // Delivery of NP / network events happens on the game's own thread, from its per-frame sysutil poll.
  g_check_orig = cellforge::hook_import("cellSysutil", "cellSysutilCheckCallback", h_sysutil_check_callback);
  if (!g_check_orig) cellforge::log("skate3_community: cellSysutilCheckCallback not found; NP events will not be delivered");
  cellforge::logf("skate3_community: %d functions hooked (%d not imported); EA hosts -> %s", g_hooked, g_missing,
                  g_cfg.server_ip.c_str());
  return 0;
}
