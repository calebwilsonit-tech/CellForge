#pragma once
// Client for the RPCN account service (the same service RPCS3 uses): TLS over TCP, small binary protocol.
// Skate 3's community server accepts the tickets RPCN issues, so signing in here is what makes the game "online".

#include <cstdint>
#include <string>
#include <vector>

namespace s3c {

struct RpcnTicket {
  bool ok = false;
  int error_code = 0;           // RPCN ErrorType when the server refused, else 0
  bool credentials_bad = false; // wrong npid / password / token: ask the user again
  std::string message;          // human readable result
  std::vector<uint8_t> ticket;  // the signed NP ticket (valid for ~15 minutes)
  std::string online_name;
};

// Connects, logs in and requests a ticket for `service_id` (e.g. "UP0006-BLUS30464_00"), then disconnects.
RpcnTicket rpcn_login_and_ticket(const std::string& host, uint16_t port, const std::string& npid, const std::string& password,
                                 const std::string& token, const std::string& service_id, const std::vector<uint8_t>& cookie);

struct RpcnCreate {
  bool ok = false;
  std::string message;
};
// Creates a new RPCN account. RPCN emails a verification token that is needed for the first login.
RpcnCreate rpcn_create_account(const std::string& host, uint16_t port, const std::string& npid, const std::string& password,
                               const std::string& email);

}  // namespace s3c
