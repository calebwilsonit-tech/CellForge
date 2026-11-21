#include "tls_client.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define SECURITY_WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <schannel.h>
#include <security.h>
#include <sspi.h>

#include <algorithm>
#include <cstring>

namespace s3c {

struct TlsClient::Impl {
  SOCKET s = INVALID_SOCKET;
  CredHandle cred{};
  CtxtHandle ctx{};
  bool have_cred = false, have_ctx = false;
  SecPkgContext_StreamSizes sizes{};
  std::vector<uint8_t> enc;    // ciphertext received but not yet decrypted
  std::vector<uint8_t> plain;  // decrypted bytes not yet consumed
  size_t plain_off = 0;

  bool send_raw(const uint8_t* p, size_t n) {
    while (n) {
      const int r = ::send(s, reinterpret_cast<const char*>(p), int(std::min<size_t>(n, 1 << 20)), 0);
      if (r <= 0) return false;
      p += r;
      n -= size_t(r);
    }
    return true;
  }

  // Receives more ciphertext (honouring the socket's receive timeout).
  bool recv_more() {
    uint8_t tmp[16384];
    const int r = ::recv(s, reinterpret_cast<char*>(tmp), sizeof tmp, 0);
    if (r <= 0) return false;
    enc.insert(enc.end(), tmp, tmp + r);
    return true;
  }

  void set_timeout(int ms) {
    DWORD t = DWORD(std::max(ms, 1));
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&t), sizeof t);
  }

  // Decrypts what is in `enc` (receiving more as needed) until at least one byte of plaintext is available.
  bool fill() {
    for (;;) {
      if (!enc.empty()) {
        SecBuffer b[4];
        b[0] = {ULONG(enc.size()), SECBUFFER_DATA, enc.data()};
        b[1] = {0, SECBUFFER_EMPTY, nullptr};
        b[2] = {0, SECBUFFER_EMPTY, nullptr};
        b[3] = {0, SECBUFFER_EMPTY, nullptr};
        SecBufferDesc d{SECBUFFER_VERSION, 4, b};
        const SECURITY_STATUS ss = DecryptMessage(&ctx, &d, 0, nullptr);
        if (ss == SEC_E_OK) {
          size_t got = 0;
          const uint8_t* extra = nullptr;
          size_t extra_len = 0;
          for (int i = 0; i < 4; ++i) {
            if (b[i].BufferType == SECBUFFER_DATA && b[i].cbBuffer) {
              plain.insert(plain.end(), static_cast<uint8_t*>(b[i].pvBuffer), static_cast<uint8_t*>(b[i].pvBuffer) + b[i].cbBuffer);
              got += b[i].cbBuffer;
            } else if (b[i].BufferType == SECBUFFER_EXTRA) {
              extra = static_cast<uint8_t*>(b[i].pvBuffer);
              extra_len = b[i].cbBuffer;
            }
          }
          if (extra_len) enc.assign(extra, extra + extra_len);
          else enc.clear();
          if (got) return true;
          continue;
        }
        if (ss != SEC_E_INCOMPLETE_MESSAGE) return false;  // incl. SEC_I_CONTEXT_EXPIRED / SEC_I_RENEGOTIATE
      }
      if (!recv_more()) return false;
    }
  }
};

TlsClient::TlsClient() : p_(new Impl) {}
TlsClient::~TlsClient() { close(); delete p_; }

bool TlsClient::is_open() const { return p_->have_ctx; }

void TlsClient::close() {
  Impl& m = *p_;
  if (m.have_ctx) {
    DWORD type = SCHANNEL_SHUTDOWN;
    SecBuffer sb{sizeof type, SECBUFFER_TOKEN, &type};
    SecBufferDesc sd{SECBUFFER_VERSION, 1, &sb};
    ApplyControlToken(&m.ctx, &sd);
    DeleteSecurityContext(&m.ctx);
    m.have_ctx = false;
  }
  if (m.have_cred) { FreeCredentialsHandle(&m.cred); m.have_cred = false; }
  if (m.s != INVALID_SOCKET) { closesocket(m.s); m.s = INVALID_SOCKET; }
  m.enc.clear();
  m.plain.clear();
  m.plain_off = 0;
}

bool TlsClient::open(const std::string& host, uint16_t port, int timeout_ms, std::string& err) {
  close();
  Impl& m = *p_;
  WSADATA wd;
  WSAStartup(MAKEWORD(2, 2), &wd);

  addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
    err = "could not resolve " + host;
    return false;
  }
  for (addrinfo* ai = res; ai; ai = ai->ai_next) {
    SOCKET s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (s == INVALID_SOCKET) continue;
    // Bounded connect: non-blocking + select.
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    ::connect(s, ai->ai_addr, int(ai->ai_addrlen));
    fd_set w;
    FD_ZERO(&w);
    FD_SET(s, &w);
    timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    if (select(0, nullptr, &w, nullptr, &tv) == 1) {
      int e = 0, l = sizeof e;
      getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&e), &l);
      if (e == 0) {
        nb = 0;
        ioctlsocket(s, FIONBIO, &nb);
        m.s = s;
        break;
      }
    }
    closesocket(s);
  }
  freeaddrinfo(res);
  if (m.s == INVALID_SOCKET) {
    err = "could not connect to " + host + ":" + std::to_string(port);
    return false;
  }
  BOOL nodelay = TRUE;
  setsockopt(m.s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof nodelay);
  m.set_timeout(timeout_ms);

  SCHANNEL_CRED sc{};
  sc.dwVersion = SCHANNEL_CRED_VERSION;
  sc.grbitEnabledProtocols = SP_PROT_TLS1_2_CLIENT;
  sc.dwFlags = SCH_CRED_MANUAL_CRED_VALIDATION | SCH_CRED_NO_DEFAULT_CREDS | SCH_CRED_NO_SERVERNAME_CHECK;
  TimeStamp ts;
  if (AcquireCredentialsHandleA(nullptr, const_cast<SEC_CHAR*>(UNISP_NAME_A), SECPKG_CRED_OUTBOUND, nullptr, &sc, nullptr, nullptr,
                                &m.cred, &ts) != SEC_E_OK) {
    err = "TLS: could not create credentials";
    close();
    return false;
  }
  m.have_cred = true;

  const DWORD flags = ISC_REQ_SEQUENCE_DETECT | ISC_REQ_REPLAY_DETECT | ISC_REQ_CONFIDENTIALITY | ISC_RET_EXTENDED_ERROR |
                      ISC_REQ_ALLOCATE_MEMORY | ISC_REQ_STREAM | ISC_REQ_MANUAL_CRED_VALIDATION;
  DWORD attrs = 0;
  SecBuffer ob{0, SECBUFFER_TOKEN, nullptr};
  SecBufferDesc od{SECBUFFER_VERSION, 1, &ob};
  SECURITY_STATUS ss = InitializeSecurityContextA(&m.cred, nullptr, const_cast<SEC_CHAR*>(host.c_str()), flags, 0, SECURITY_NATIVE_DREP,
                                                  nullptr, 0, &m.ctx, &od, &attrs, &ts);
  if (ss != SEC_I_CONTINUE_NEEDED) {
    err = "TLS: handshake could not start (0x" + std::to_string(unsigned(ss)) + ")";
    close();
    return false;
  }
  m.have_ctx = true;
  bool sent = ob.cbBuffer && m.send_raw(static_cast<uint8_t*>(ob.pvBuffer), ob.cbBuffer);
  if (ob.pvBuffer) FreeContextBuffer(ob.pvBuffer);
  if (!sent) { err = "TLS: send failed during handshake"; close(); return false; }

  bool need_more = true;
  for (;;) {
    if (need_more && !m.recv_more()) { err = "TLS: connection closed during handshake"; close(); return false; }
    SecBuffer ib[2];
    ib[0] = {ULONG(m.enc.size()), SECBUFFER_TOKEN, m.enc.data()};
    ib[1] = {0, SECBUFFER_EMPTY, nullptr};
    SecBufferDesc id{SECBUFFER_VERSION, 2, ib};
    ob = {0, SECBUFFER_TOKEN, nullptr};
    ss = InitializeSecurityContextA(&m.cred, &m.ctx, const_cast<SEC_CHAR*>(host.c_str()), flags, 0, SECURITY_NATIVE_DREP, &id, 0, nullptr,
                                    &od, &attrs, &ts);
    if (ob.cbBuffer && ob.pvBuffer) {
      sent = m.send_raw(static_cast<uint8_t*>(ob.pvBuffer), ob.cbBuffer);
      FreeContextBuffer(ob.pvBuffer);
      if (!sent) { err = "TLS: send failed during handshake"; close(); return false; }
    }
    if (ss == SEC_E_INCOMPLETE_MESSAGE) { need_more = true; continue; }
    if (ib[1].BufferType == SECBUFFER_EXTRA && ib[1].cbBuffer) m.enc.erase(m.enc.begin(), m.enc.end() - ib[1].cbBuffer);
    else m.enc.clear();
    if (ss == SEC_E_OK) break;
    if (ss == SEC_I_CONTINUE_NEEDED) { need_more = m.enc.empty(); continue; }
    char b[64];
    std::snprintf(b, sizeof b, "TLS: handshake failed (0x%08X)", unsigned(ss));
    err = b;
    close();
    return false;
  }
  if (QueryContextAttributes(&m.ctx, SECPKG_ATTR_STREAM_SIZES, &m.sizes) != SEC_E_OK) {
    err = "TLS: could not query stream sizes";
    close();
    return false;
  }
  return true;
}

bool TlsClient::write(const void* data, size_t len) {
  Impl& m = *p_;
  if (!m.have_ctx) return false;
  const uint8_t* src = static_cast<const uint8_t*>(data);
  while (len) {
    const size_t chunk = std::min<size_t>(len, m.sizes.cbMaximumMessage);
    std::vector<uint8_t> buf(m.sizes.cbHeader + chunk + m.sizes.cbTrailer);
    std::memcpy(buf.data() + m.sizes.cbHeader, src, chunk);
    SecBuffer b[4];
    b[0] = {m.sizes.cbHeader, SECBUFFER_STREAM_HEADER, buf.data()};
    b[1] = {ULONG(chunk), SECBUFFER_DATA, buf.data() + m.sizes.cbHeader};
    b[2] = {m.sizes.cbTrailer, SECBUFFER_STREAM_TRAILER, buf.data() + m.sizes.cbHeader + chunk};
    b[3] = {0, SECBUFFER_EMPTY, nullptr};
    SecBufferDesc d{SECBUFFER_VERSION, 4, b};
    if (EncryptMessage(&m.ctx, 0, &d, 0) != SEC_E_OK) return false;
    if (!m.send_raw(buf.data(), b[0].cbBuffer + b[1].cbBuffer + b[2].cbBuffer)) return false;
    src += chunk;
    len -= chunk;
  }
  return true;
}

bool TlsClient::read_exact(void* out, size_t len, int timeout_ms) {
  Impl& m = *p_;
  if (!m.have_ctx) return false;
  m.set_timeout(timeout_ms);
  uint8_t* dst = static_cast<uint8_t*>(out);
  while (len) {
    if (m.plain_off >= m.plain.size()) {
      m.plain.clear();
      m.plain_off = 0;
      if (!m.fill()) return false;
    }
    const size_t n = std::min(len, m.plain.size() - m.plain_off);
    std::memcpy(dst, m.plain.data() + m.plain_off, n);
    m.plain_off += n;
    dst += n;
    len -= n;
  }
  return true;
}

}  // namespace s3c
