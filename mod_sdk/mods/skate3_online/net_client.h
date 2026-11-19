#pragma once
// Winsock2 UDP P2P / Relay client with background I/O thread and peer interpolation.

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "protocol.h"

namespace s3net {

struct PeerSnapshot {
  uint8_t peer_id = 0;
  std::string name;
  SkaterState prev_state{};
  SkaterState curr_state{};
  uint64_t prev_recv_us = 0;
  uint64_t curr_recv_us = 0;
  uint32_t last_seq = 0;
  uint32_t rtt_ms = 0;
  bool active = false;
};

struct ClientConfig {
  std::string mode = "relay";        // "host", "join", or "relay"
  std::string host = "127.0.0.1";    // Host or relay IP/hostname
  uint16_t port = kDefaultPort;
  std::string room = "SK8A";         // Room code (used in relay mode)
  std::string player_name = "Skater";
  uint16_t tick_hz = 60;
};

class NetClient {
 public:
  using LogFn = std::function<void(const std::string&)>;

  NetClient() = default;
  ~NetClient() { stop(); }

  bool start(const ClientConfig& cfg, LogFn log = nullptr);
  void stop();

  // Thread-safe: call from the game loop to publish local skater state.
  void set_local_state(const SkaterState& st);

  // Thread-safe: returns interpolated remote peers at the current timestamp.
  std::vector<PeerSnapshot> get_remote_peers() const;

  // Sends a short text/event packet (e.g. session marker or trick callout).
  void send_event(uint16_t kind, const std::string& payload);

  bool running() const { return running_.load(); }
  uint8_t local_peer_id() const { return peer_id_.load(); }
  uint32_t rtt_ms() const { return rtt_ms_.load(); }
  uint64_t packets_sent() const { return tx_count_.load(); }
  uint64_t packets_recv() const { return rx_count_.load(); }

  // Linearly interpolates between prev_state and curr_state for smooth rendering.
  static SkaterState interpolate(const PeerSnapshot& peer, uint64_t now_us);

 private:
  void io_loop();
  void handle_packet(const uint8_t* data, size_t len, uint32_t from_ip, uint16_t from_port, uint64_t now_us);
  void send_raw(const void* data, size_t len, uint32_t dst_ip, uint16_t dst_port);
  void logf(const char* fmt, ...);

  ClientConfig cfg_;
  LogFn log_;
  uintptr_t sock_ = ~uintptr_t(0);
  uint32_t target_ip_ = 0;
  uint16_t target_port_ = 0;

  std::atomic<bool> running_{false};
  std::atomic<uint8_t> peer_id_{0};
  std::atomic<uint32_t> seq_{1};
  std::atomic<uint32_t> rtt_ms_{0};
  std::atomic<uint64_t> tx_count_{0};
  std::atomic<uint64_t> rx_count_{0};

  mutable std::mutex state_mu_;
  SkaterState local_state_{};
  bool local_state_valid_ = false;

  struct Endpoint {
    uint32_t ip = 0;
    uint16_t port = 0;
    uint8_t peer_id = 0;
    uint64_t last_seen_us = 0;
  };

  mutable std::mutex peers_mu_;
  std::unordered_map<uint8_t, PeerSnapshot> peers_;
  std::vector<Endpoint> host_clients_;  // Used when mode == "host"
  uint8_t next_peer_id_ = 2;

  std::thread worker_;
};

}  // namespace s3net
