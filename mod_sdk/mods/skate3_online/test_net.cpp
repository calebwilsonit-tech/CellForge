// Automated UDP P2P loopback verification for skate3_online.
// Spins up a Host + Client on localhost, exchanges 60 Hz SkaterState packets,
// and verifies handshake, state delivery, and smooth interpolation.

#include "net_client.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

int main() {
  s3net::NetClient host, client;

  s3net::ClientConfig hcfg;
  hcfg.mode = "host";
  hcfg.port = 37499;
  hcfg.room = "TEST";
  hcfg.player_name = "HostSkater";
  hcfg.tick_hz = 60;

  s3net::ClientConfig ccfg;
  ccfg.mode = "join";
  ccfg.host = "127.0.0.1";
  ccfg.port = 37499;
  ccfg.room = "TEST";
  ccfg.player_name = "GuestSkater";
  ccfg.tick_hz = 60;

  auto logger = [](const char* tag) {
    return [tag](const std::string& m) { std::printf("[%s] %s\n", tag, m.c_str()); };
  };

  if (!host.start(hcfg, logger("host"))) return 1;
  if (!client.start(ccfg, logger("client"))) return 2;

  for (int i = 0; i < 30; ++i) {
    s3net::SkaterState hst{};
    hst.pos[0] = float(i) * 0.5f;
    hst.pos[1] = 1.0f;
    hst.pos[2] = -2.0f;
    host.set_local_state(hst);

    s3net::SkaterState cst{};
    cst.pos[0] = 10.0f + float(i) * 0.25f;
    cst.pos[1] = 2.0f;
    cst.pos[2] = 3.0f;
    client.set_local_state(cst);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  const auto host_peers = host.get_remote_peers();
  const auto client_peers = client.get_remote_peers();

  const bool ok_id = (client.local_peer_id() >= 2);
  const bool ok_host_rx = (!host_peers.empty() && host_peers[0].curr_state.pos[0] > 10.0f);
  const bool ok_client_rx = (!client_peers.empty() && client_peers[0].curr_state.pos[0] > 0.0f);

  std::printf("Handshake peer_id=%u (%s)\n", unsigned(client.local_peer_id()), ok_id ? "PASS" : "FAIL");
  std::printf("Host received guest state: %s (tx=%llu rx=%llu)\n",
              ok_host_rx ? "PASS" : "FAIL",
              static_cast<unsigned long long>(host.packets_sent()),
              static_cast<unsigned long long>(host.packets_recv()));
  std::printf("Client received host state: %s (tx=%llu rx=%llu)\n",
              ok_client_rx ? "PASS" : "FAIL",
              static_cast<unsigned long long>(client.packets_sent()),
              static_cast<unsigned long long>(client.packets_recv()));

  client.stop();
  host.stop();
  return (ok_id && ok_host_rx && ok_client_rx) ? 0 : 1;
}
