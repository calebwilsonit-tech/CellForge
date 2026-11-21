// Connectivity check for the RPCN client: TLS handshake, server hello, and one login attempt with a made-up
// account (expected answer: "unknown username"). Usage: rpcn_probe [host] [port]
#include <cstdio>
#include <cstdlib>
#include "rpcn.h"

int main(int argc, char** argv) {
  const std::string host = argc > 1 ? argv[1] : "np.rpcs3.net";
  const uint16_t port = argc > 2 ? uint16_t(std::atoi(argv[2])) : 31313;
  const auto r = s3c::rpcn_login_and_ticket(host, port, "cellforge_probe_nouser", "x", "", "UP0006-BLUS30464_00", {});
  std::printf("ok=%d error_code=%d credentials_bad=%d message=%s\n", r.ok, r.error_code, r.credentials_bad, r.message.c_str());
  return r.ok ? 0 : (r.error_code ? 0 : 1);
}
