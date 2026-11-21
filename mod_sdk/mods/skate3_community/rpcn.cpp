#include "rpcn.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>

#include "tls_client.h"

namespace s3c {
namespace {

constexpr size_t kHeader = 15;
constexpr uint32_t kProtocolVersion = 34;
constexpr uint32_t kMaxPacket = 0x800000;
enum : uint8_t { kPktRequest = 0, kPktReply = 1, kPktNotification = 2, kPktServerInfo = 3 };
enum : uint16_t { kCmdLogin = 0, kCmdTerminate = 1, kCmdCreate = 2, kCmdRequestTicket = 28 };

std::atomic<uint64_t> g_packet_id{0x100000001ull};

void put_le(std::vector<uint8_t>& v, uint64_t x, int bytes) {
  for (int i = 0; i < bytes; ++i) v.push_back(uint8_t(x >> (8 * i)));
}
void put_cstr(std::vector<uint8_t>& v, const std::string& s) {
  v.insert(v.end(), s.begin(), s.end());
  v.push_back(0);
}

uint64_t get_le(const uint8_t* p, int bytes) {
  uint64_t x = 0;
  for (int i = 0; i < bytes; ++i) x |= uint64_t(p[i]) << (8 * i);
  return x;
}

std::vector<uint8_t> forge(uint16_t cmd, uint64_t id, const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> p;
  p.push_back(kPktRequest);
  put_le(p, cmd, 2);
  put_le(p, kHeader + payload.size(), 4);
  put_le(p, id, 8);
  p.insert(p.end(), payload.begin(), payload.end());
  return p;
}

struct Packet {
  uint8_t type = 0;
  uint16_t cmd = 0;
  uint64_t id = 0;
  std::vector<uint8_t> payload;
};

bool read_packet(TlsClient& tls, Packet& out, int timeout_ms) {
  uint8_t h[kHeader];
  if (!tls.read_exact(h, kHeader, timeout_ms)) return false;
  const uint32_t size = uint32_t(get_le(h + 3, 4));
  if (size < kHeader || size > kMaxPacket) return false;
  out.type = h[0];
  out.cmd = uint16_t(get_le(h + 1, 2));
  out.id = get_le(h + 7, 8);
  out.payload.resize(size - kHeader);
  return out.payload.empty() || tls.read_exact(out.payload.data(), out.payload.size(), timeout_ms);
}

// Sends a request and waits for its reply (notifications and unrelated packets are skipped).
bool transact(TlsClient& tls, uint16_t cmd, const std::vector<uint8_t>& payload, std::vector<uint8_t>& reply, int timeout_ms) {
  const uint64_t id = g_packet_id.fetch_add(1);
  const auto bytes = forge(cmd, id, payload);
  if (!tls.write(bytes.data(), bytes.size())) return false;
  for (int guard = 0; guard < 256; ++guard) {
    Packet pk;
    if (!read_packet(tls, pk, timeout_ms)) return false;
    if (pk.type == kPktReply && pk.id == id) {
      reply = std::move(pk.payload);
      return true;
    }
  }
  return false;
}

// Opens the connection and checks the server's hello (protocol version).
bool connect_rpcn(TlsClient& tls, const std::string& host, uint16_t port, std::string& err) {
  if (!tls.open(host, port, 10000, err)) return false;
  Packet info;
  if (!read_packet(tls, info, 10000) || info.type != kPktServerInfo || info.payload.size() < 4) {
    err = "the RPCN server did not send its hello";
    return false;
  }
  const uint32_t ver = uint32_t(get_le(info.payload.data(), 4));
  if (ver != kProtocolVersion) {
    char b[96];
    std::snprintf(b, sizeof b, "RPCN protocol version %u is not supported (expected %u)", ver, kProtocolVersion);
    err = b;
    return false;
  }
  return true;
}

const char* error_text(int code) {
  switch (code) {
    case 0: return "ok";
    case 1: return "the server did not understand the request";
    case 2: return "invalid request";
    case 3: return "invalid input";
    case 4: return "too soon, please wait before trying again";
    case 5: return "login error";
    case 6: return "this account is already signed in somewhere else (RPCS3 or another copy of the game)";
    case 7: return "unknown username";
    case 8: return "wrong password";
    case 9: return "wrong or missing email token";
    case 10: return "account creation failed";
    case 11: return "that username is already taken";
    case 12: return "that email provider is not allowed";
    case 13: return "that email address is already registered";
    case 24: return "the server had a database problem";
    case 25: return "the server could not send the email";
    default: return "the server refused the request";
  }
}

}  // namespace

RpcnTicket rpcn_login_and_ticket(const std::string& host, uint16_t port, const std::string& npid, const std::string& password,
                                 const std::string& token, const std::string& service_id, const std::vector<uint8_t>& cookie) {
  RpcnTicket r;
  TlsClient tls;
  std::string err;
  if (!connect_rpcn(tls, host, port, err)) {
    r.message = err;
    return r;
  }

  std::vector<uint8_t> payload, reply;
  put_cstr(payload, npid);
  put_cstr(payload, password);
  put_cstr(payload, token);
  if (!transact(tls, kCmdLogin, payload, reply, 15000) || reply.empty()) {
    r.message = "no answer to the login request";
    return r;
  }
  if (reply[0] != 0) {
    r.error_code = reply[0];
    r.credentials_bad = reply[0] == 7 || reply[0] == 8 || reply[0] == 9;
    r.message = error_text(reply[0]);
    return r;
  }
  // Reply: online_name\0 avatar_url\0 user_id(s64)... (friend lists follow; not needed here)
  {
    size_t i = 1;
    while (i < reply.size() && reply[i]) r.online_name.push_back(char(reply[i++]));
  }

  if (service_id.empty()) {  // login check only
    r.ok = true;
    r.message = "signed in";
    const auto bye = forge(kCmdTerminate, g_packet_id.fetch_add(1), {});
    tls.write(bye.data(), bye.size());
    return r;
  }

  payload.clear();
  put_cstr(payload, service_id);
  put_le(payload, cookie.size(), 4);
  payload.insert(payload.end(), cookie.begin(), cookie.end());
  if (!transact(tls, kCmdRequestTicket, payload, reply, 15000) || reply.empty()) {
    r.message = "no answer to the ticket request";
    return r;
  }
  if (reply[0] != 0) {
    r.error_code = reply[0];
    r.message = error_text(reply[0]);
    return r;
  }
  if (reply.size() < 5) {
    r.message = "the ticket reply was too short";
    return r;
  }
  const uint32_t len = uint32_t(get_le(reply.data() + 1, 4));
  if (len == 0 || size_t(len) + 5 > reply.size()) {
    r.message = "the ticket reply was malformed";
    return r;
  }
  r.ticket.assign(reply.begin() + 5, reply.begin() + 5 + len);
  r.ok = true;
  r.message = "signed in";

  // Polite goodbye (the server closes the session either way).
  const auto bye = forge(kCmdTerminate, g_packet_id.fetch_add(1), {});
  tls.write(bye.data(), bye.size());
  return r;
}

RpcnCreate rpcn_create_account(const std::string& host, uint16_t port, const std::string& npid, const std::string& password,
                               const std::string& email) {
  RpcnCreate r;
  // The server rejects bad input by hanging up, so check it here and say what is wrong.
  bool name_ok = npid.size() >= 3 && npid.size() <= 16;
  for (char ch : npid) name_ok = name_ok && (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_');
  if (!name_ok) {
    r.message = "the username must be 3-16 characters: letters, digits, - or _";
    return r;
  }
  if (password.empty()) {
    r.message = "please enter a password";
    return r;
  }
  const size_t at = email.find('@');
  if (at == std::string::npos || at == 0 || at + 3 > email.size() || email.find(' ') != std::string::npos ||
      email.find('.', at) == std::string::npos) {
    r.message = "please enter a valid email address";
    return r;
  }
  TlsClient tls;
  std::string err;
  if (!connect_rpcn(tls, host, port, err)) {
    r.message = err;
    return r;
  }
  std::vector<uint8_t> payload, reply;
  put_cstr(payload, npid);
  put_cstr(payload, password);
  put_cstr(payload, npid);  // online name
  put_cstr(payload, "https://rpcs3.net/cdn/netplay/DefaultAvatar.png");  // avatar url (must not be empty)
  put_cstr(payload, email);
  if (!transact(tls, kCmdCreate, payload, reply, 20000) || reply.empty()) {
    r.message = "no answer to the account request";
    return r;
  }
  if (reply[0] != 0) {
    r.message = error_text(reply[0]);
    if (reply[0] == 3) r.message = "invalid input: the name must be 3-16 letters, digits, - or _, and the email must be valid";
    return r;
  }
  r.ok = true;
  r.message = "account created; check your email for the verification token";
  return r;
}

}  // namespace s3c
