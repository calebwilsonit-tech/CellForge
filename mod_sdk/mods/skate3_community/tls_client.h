#pragma once
// Minimal blocking TLS 1.2 client on top of Winsock + Windows SChannel (no third-party libraries).
// RPCN servers use self-signed certificates, so the certificate chain is not validated (same as RPCS3's client).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace s3c {

class TlsClient {
 public:
  TlsClient();
  ~TlsClient();
  TlsClient(const TlsClient&) = delete;
  TlsClient& operator=(const TlsClient&) = delete;

  // Resolves, connects and runs the TLS handshake. On failure returns false and fills `err`.
  bool open(const std::string& host, uint16_t port, int timeout_ms, std::string& err);
  bool write(const void* data, size_t len);
  // Reads exactly `len` decrypted bytes. False on error, timeout or when the peer closed.
  bool read_exact(void* out, size_t len, int timeout_ms);
  void close();
  bool is_open() const;

 private:
  struct Impl;
  Impl* p_;
};

}  // namespace s3c
