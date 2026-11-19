#include "net_client.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace s3net {
namespace {

uint64_t steady_us() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint32_t steady_ms() { return static_cast<uint32_t>(steady_us() / 1000ull); }

bool ensure_wsa() {
  static const bool ok = [] {
    WSADATA wsa{};
    return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
  }();
  return ok;
}

bool resolve_ipv4(const std::string& host, uint32_t& out_ip) {
  in_addr addr{};
  if (inet_pton(AF_INET, host.c_str(), &addr) == 1) {
    out_ip = addr.s_addr;
    return true;
  }
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo* res = nullptr;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return false;
  out_ip = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr.s_addr;
  freeaddrinfo(res);
  return true;
}

}  // namespace

void NetClient::logf(const char* fmt, ...) {
  if (!log_) return;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  log_(buf);
}

bool NetClient::start(const ClientConfig& cfg, LogFn log) {
  stop();
  cfg_ = cfg;
  log_ = std::move(log);
  if (cfg_.tick_hz < 10) cfg_.tick_hz = 10;
  if (cfg_.tick_hz > 120) cfg_.tick_hz = 120;

  if (!ensure_wsa()) {
    logf("WSAStartup failed");
    return false;
  }

  SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (s == INVALID_SOCKET) {
    logf("socket() failed (%d)", WSAGetLastError());
    return false;
  }

  u_long nonblock = 1;
  ioctlsocket(s, FIONBIO, &nonblock);

  sockaddr_in bind_addr{};
  bind_addr.sin_family = AF_INET;
  bind_addr.sin_addr.s_addr = INADDR_ANY;
  bind_addr.sin_port = (cfg_.mode == "host") ? htons(cfg_.port) : 0;
  if (bind(s, reinterpret_cast<sockaddr*>(&bind_addr), sizeof(bind_addr)) != 0) {
    logf("bind() failed on port %u (%d)", unsigned(cfg_.port), WSAGetLastError());
    closesocket(s);
    return false;
  }

  if (cfg_.mode == "host") {
    peer_id_ = 1;
    target_ip_ = 0;
    target_port_ = 0;
    logf("hosting P2P session on UDP port %u (room=%s, name=%s, %u Hz)",
         unsigned(cfg_.port), cfg_.room.c_str(), cfg_.player_name.c_str(), unsigned(cfg_.tick_hz));
  } else {
    if (!resolve_ipv4(cfg_.host, target_ip_)) {
      logf("could not resolve host '%s'", cfg_.host.c_str());
      closesocket(s);
      return false;
    }
    target_port_ = htons(cfg_.port);
    peer_id_ = 0;
    logf("connecting (%s) to %s:%u (room=%s, name=%s, %u Hz)",
         cfg_.mode.c_str(), cfg_.host.c_str(), unsigned(cfg_.port),
         cfg_.room.c_str(), cfg_.player_name.c_str(), unsigned(cfg_.tick_hz));
  }

  sock_ = static_cast<uintptr_t>(s);
  running_ = true;
  worker_ = std::thread(&NetClient::io_loop, this);
  return true;
}

void NetClient::stop() {
  if (!running_.exchange(false)) return;

  // Send graceful Leave packet if connected.
  if (sock_ != ~uintptr_t(0) && cfg_.mode != "host" && target_ip_ != 0) {
    Header h{};
    h.type = static_cast<uint8_t>(MsgType::Leave);
    h.peer_id = peer_id_.load();
    h.seq = seq_.fetch_add(1);
    h.timestamp_ms = steady_ms();
    send_raw(&h, sizeof(h), target_ip_, target_port_);
  }

  if (worker_.joinable()) worker_.join();
  if (sock_ != ~uintptr_t(0)) {
    closesocket(static_cast<SOCKET>(sock_));
    sock_ = ~uintptr_t(0);
  }
  {
    std::lock_guard<std::mutex> lk(peers_mu_);
    peers_.clear();
    host_clients_.clear();
  }
  peer_id_ = 0;
}

void NetClient::set_event_callback(EventFn cb) {
  std::lock_guard<std::mutex> lk(event_mu_);
  event_cb_ = std::move(cb);
}

void NetClient::set_local_state(const SkaterState& st) {
  std::lock_guard<std::mutex> lk(state_mu_);
  local_state_ = st;
  local_state_valid_ = true;
}

std::vector<PeerSnapshot> NetClient::get_remote_peers() const {
  std::lock_guard<std::mutex> lk(peers_mu_);
  std::vector<PeerSnapshot> out;
  out.reserve(peers_.size());
  for (const auto& kv : peers_) {
    if (kv.second.active) out.push_back(kv.second);
  }
  return out;
}

void NetClient::send_event(uint16_t kind, const std::string& payload) {
  if (!running_.load()) return;
  EventPacket pkt{};
  pkt.hdr.type = static_cast<uint8_t>(MsgType::Event);
  pkt.hdr.peer_id = peer_id_.load();
  pkt.hdr.seq = seq_.fetch_add(1);
  pkt.hdr.timestamp_ms = steady_ms();
  copy_str(pkt.room, kRoomCodeLen, cfg_.room.c_str());
  pkt.event_kind = kind;
  const size_t n = std::min(payload.size(), sizeof(pkt.payload) - 1);
  pkt.payload_len = static_cast<uint16_t>(n);
  std::memcpy(pkt.payload, payload.data(), n);

  if (cfg_.mode == "host") {
    std::lock_guard<std::mutex> lk(peers_mu_);
    for (const auto& ep : host_clients_) send_raw(&pkt, sizeof(pkt), ep.ip, ep.port);
  } else if (target_ip_ != 0) {
    send_raw(&pkt, sizeof(pkt), target_ip_, target_port_);
  }
}

void NetClient::send_raw(const void* data, size_t len, uint32_t dst_ip, uint16_t dst_port) {
  if (sock_ == ~uintptr_t(0)) return;
  sockaddr_in dst{};
  dst.sin_family = AF_INET;
  dst.sin_addr.s_addr = dst_ip;
  dst.sin_port = dst_port;
  const int sent = sendto(static_cast<SOCKET>(sock_), reinterpret_cast<const char*>(data),
                          static_cast<int>(len), 0, reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
  if (sent > 0) tx_count_.fetch_add(1);
}

void NetClient::io_loop() {
  const uint64_t tick_us = 1000000ull / cfg_.tick_hz;
  uint64_t last_state_us = 0;
  uint64_t last_hello_us = 0;
  uint64_t last_ping_us = 0;
  uint64_t last_info_us = 0;

  uint8_t buf[1024];
  while (running_.load()) {
    const uint64_t now = steady_us();

    // 1. Drain all incoming UDP datagrams.
    for (;;) {
      sockaddr_in from{};
      int from_len = sizeof(from);
      const int n = recvfrom(static_cast<SOCKET>(sock_), reinterpret_cast<char*>(buf),
                             sizeof(buf), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
      if (n <= 0) break;
      rx_count_.fetch_add(1);
      handle_packet(buf, static_cast<size_t>(n), from.sin_addr.s_addr, from.sin_port, now);
    }

    // 2. Handshake / keepalive Hello (join or relay mode until assigned peer_id).
    if (cfg_.mode != "host" && peer_id_.load() == 0 && (now - last_hello_us) >= 500000ull) {
      HelloPacket h{};
      h.hdr.type = static_cast<uint8_t>(MsgType::Hello);
      h.hdr.seq = seq_.fetch_add(1);
      h.hdr.timestamp_ms = steady_ms();
      copy_str(h.room, kRoomCodeLen, cfg_.room.c_str());
      copy_str(h.name, kNameLen, cfg_.player_name.c_str());
      send_raw(&h, sizeof(h), target_ip_, target_port_);
      last_hello_us = now;
    }

    // 3. Periodic Ping every 2s for RTT measurement.
    if (cfg_.mode != "host" && peer_id_.load() != 0 && (now - last_ping_us) >= 2000000ull) {
      PingPacket p{};
      p.hdr.type = static_cast<uint8_t>(MsgType::Ping);
      p.hdr.peer_id = peer_id_.load();
      p.hdr.seq = seq_.fetch_add(1);
      p.hdr.timestamp_ms = steady_ms();
      p.send_time_us = now;
      send_raw(&p, sizeof(p), target_ip_, target_port_);
      last_ping_us = now;
    }

    // 3b. Broadcast local player name periodically (every 4s) so late joiners learn our name.
    if (peer_id_.load() != 0 && (now - last_info_us) >= 4000000ull) {
      send_event(static_cast<uint16_t>(EventKind::PeerInfo), cfg_.player_name);
      last_info_us = now;
    }

    // 4. Send local skater state at configured tick_hz.
    if ((now - last_state_us) >= tick_us && peer_id_.load() != 0) {
      SkaterState st{};
      bool valid = false;
      {
        std::lock_guard<std::mutex> lk(state_mu_);
        st = local_state_;
        valid = local_state_valid_;
      }
      if (valid) {
        StatePacket sp{};
        sp.hdr.type = static_cast<uint8_t>(MsgType::State);
        sp.hdr.peer_id = peer_id_.load();
        sp.hdr.seq = seq_.fetch_add(1);
        sp.hdr.timestamp_ms = steady_ms();
        copy_str(sp.room, kRoomCodeLen, cfg_.room.c_str());
        sp.state = st;

        if (cfg_.mode == "host") {
          std::lock_guard<std::mutex> lk(peers_mu_);
          for (const auto& ep : host_clients_) send_raw(&sp, sizeof(sp), ep.ip, ep.port);
        } else {
          send_raw(&sp, sizeof(sp), target_ip_, target_port_);
        }
      }
      last_state_us = now;
    }

    // 5. Prune timed-out peers (no packets for >10s).
    {
      std::lock_guard<std::mutex> lk(peers_mu_);
      for (auto& kv : peers_) {
        if (kv.second.active && kv.second.curr_recv_us > 0 && (now - kv.second.curr_recv_us) > 10000000ull) {
          kv.second.active = false;
          logf("peer %u (%s) timed out", unsigned(kv.first), kv.second.name.c_str());
        }
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void NetClient::handle_packet(const uint8_t* data, size_t len, uint32_t from_ip, uint16_t from_port, uint64_t now_us) {
  if (len < sizeof(Header)) return;
  Header hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));
  if (hdr.magic != kMagic || hdr.version != kProtocolVersion) return;

  const auto type = static_cast<MsgType>(hdr.type);

  if (type == MsgType::Hello && cfg_.mode == "host" && len >= sizeof(HelloPacket)) {
    HelloPacket hp{};
    std::memcpy(&hp, data, sizeof(hp));
    uint8_t assigned = 0;
    {
      std::lock_guard<std::mutex> lk(peers_mu_);
      for (auto& ep : host_clients_) {
        if (ep.ip == from_ip && ep.port == from_port) {
          assigned = ep.peer_id;
          ep.last_seen_us = now_us;
          break;
        }
      }
      if (assigned == 0 && host_clients_.size() < kMaxPlayers - 1) {
        assigned = next_peer_id_++;
        host_clients_.push_back(Endpoint{from_ip, from_port, assigned, now_us});
        PeerSnapshot& ps = peers_[assigned];
        ps.peer_id = assigned;
        ps.name = std::string(hp.name, strnlen(hp.name, kNameLen));
        ps.curr_recv_us = now_us;
        ps.active = true;
        logf("player '%s' joined (peer_id=%u)", ps.name.c_str(), unsigned(assigned));
      }
    }
    if (assigned != 0) {
      WelcomePacket wp{};
      wp.hdr.type = static_cast<uint8_t>(MsgType::Welcome);
      wp.hdr.peer_id = 1;
      wp.hdr.seq = seq_.fetch_add(1);
      wp.hdr.timestamp_ms = steady_ms();
      wp.assigned_peer_id = assigned;
      wp.player_count = static_cast<uint8_t>(host_clients_.size() + 1);
      wp.tick_hz = cfg_.tick_hz;
      send_raw(&wp, sizeof(wp), from_ip, from_port);

      // Immediately share host's name with the newly joined client.
      EventPacket info{};
      info.hdr.type = static_cast<uint8_t>(MsgType::Event);
      info.hdr.peer_id = 1;
      info.hdr.seq = seq_.fetch_add(1);
      info.hdr.timestamp_ms = steady_ms();
      copy_str(info.room, kRoomCodeLen, cfg_.room.c_str());
      info.event_kind = static_cast<uint16_t>(EventKind::PeerInfo);
      const size_t n = std::min(cfg_.player_name.size(), sizeof(info.payload) - 1);
      info.payload_len = static_cast<uint16_t>(n);
      std::memcpy(info.payload, cfg_.player_name.data(), n);
      send_raw(&info, sizeof(info), from_ip, from_port);
    }
    return;
  }

  if (type == MsgType::Welcome && len >= sizeof(WelcomePacket)) {
    WelcomePacket wp{};
    std::memcpy(&wp, data, sizeof(wp));
    if (peer_id_.load() == 0) {
      peer_id_ = wp.assigned_peer_id;
      logf("connected! assigned peer_id=%u (players=%u, %u Hz)",
           unsigned(wp.assigned_peer_id), unsigned(wp.player_count), unsigned(wp.tick_hz));
      send_event(static_cast<uint16_t>(EventKind::PeerInfo), cfg_.player_name);
    }
    return;
  }

  if (type == MsgType::Ping && len >= sizeof(PingPacket)) {
    PingPacket pp{};
    std::memcpy(&pp, data, sizeof(pp));
    pp.hdr.type = static_cast<uint8_t>(MsgType::Pong);
    pp.hdr.peer_id = peer_id_.load();
    send_raw(&pp, sizeof(pp), from_ip, from_port);
    return;
  }

  if (type == MsgType::Pong && len >= sizeof(PingPacket)) {
    PingPacket pp{};
    std::memcpy(&pp, data, sizeof(pp));
    if (now_us >= pp.send_time_us) {
      rtt_ms_ = static_cast<uint32_t>((now_us - pp.send_time_us) / 1000ull);
    }
    return;
  }

  if (type == MsgType::State && len >= sizeof(StatePacket)) {
    StatePacket sp{};
    std::memcpy(&sp, data, sizeof(sp));
    if (sp.hdr.peer_id == 0 || sp.hdr.peer_id == peer_id_.load()) return;

    {
      std::lock_guard<std::mutex> lk(peers_mu_);
      PeerSnapshot& ps = peers_[sp.hdr.peer_id];
      if (ps.active && sp.hdr.seq <= ps.last_seq) return;  // drop out-of-order UDP
      ps.peer_id = sp.hdr.peer_id;
      ps.prev_state = ps.active ? ps.curr_state : sp.state;
      ps.prev_recv_us = ps.active ? ps.curr_recv_us : now_us;
      ps.curr_state = sp.state;
      ps.curr_recv_us = now_us;
      ps.last_seq = sp.hdr.seq;
      ps.active = true;

      // In host mode, forward each client's state to all other clients.
      if (cfg_.mode == "host") {
        for (const auto& ep : host_clients_) {
          if (ep.peer_id != sp.hdr.peer_id) send_raw(&sp, sizeof(sp), ep.ip, ep.port);
        }
      }
    }
    return;
  }

  if (type == MsgType::Event && len >= sizeof(EventPacket)) {
    EventPacket ep{};
    std::memcpy(&ep, data, sizeof(ep));
    const size_t plen = std::min<size_t>(ep.payload_len, sizeof(ep.payload) - 1);
    std::string msg(ep.payload, plen);

    if (ep.event_kind == static_cast<uint16_t>(EventKind::PeerInfo) && ep.hdr.peer_id != 0 &&
        ep.hdr.peer_id != peer_id_.load()) {
      std::lock_guard<std::mutex> lk(peers_mu_);
      PeerSnapshot& ps = peers_[ep.hdr.peer_id];
      ps.peer_id = ep.hdr.peer_id;
      if (!msg.empty()) ps.name = msg;
      ps.curr_recv_us = now_us;
      ps.active = true;
    } else {
      logf("event(kind=%u, from=%u): %s", unsigned(ep.event_kind), unsigned(ep.hdr.peer_id), msg.c_str());
    }

    {
      EventFn cb;
      {
        std::lock_guard<std::mutex> lk(event_mu_);
        cb = event_cb_;
      }
      if (cb) cb(ep.hdr.peer_id, ep.event_kind, msg);
    }

    if (cfg_.mode == "host") {
      std::lock_guard<std::mutex> lk(peers_mu_);
      for (const auto& c : host_clients_) {
        if (c.peer_id != ep.hdr.peer_id) send_raw(&ep, sizeof(ep), c.ip, c.port);
      }
    }
    return;
  }

  if (type == MsgType::Leave) {
    std::lock_guard<std::mutex> lk(peers_mu_);
    auto it = peers_.find(hdr.peer_id);
    if (it != peers_.end()) {
      it->second.active = false;
      logf("peer %u (%s) left", unsigned(hdr.peer_id), it->second.name.c_str());
    }
  }
}

namespace {
void nlerp_quat(const float a[4], const float b[4], float t, float out[4]) {
  float dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const float sign = (dot < 0.0f) ? -1.0f : 1.0f;
  float len2 = 0.0f;
  for (int i = 0; i < 4; ++i) {
    out[i] = a[i] + t * (sign * b[i] - a[i]);
    len2 += out[i] * out[i];
  }
  if (len2 > 1e-8f) {
    const float inv = 1.0f / std::sqrt(len2);
    for (int i = 0; i < 4; ++i) out[i] *= inv;
  } else {
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
  }
}
}  // namespace

SkaterState NetClient::interpolate(const PeerSnapshot& peer, uint64_t now_us) {
  if (!peer.active || peer.curr_recv_us <= peer.prev_recv_us) return peer.curr_state;
  const double dt = static_cast<double>(peer.curr_recv_us - peer.prev_recv_us);
  const double elapsed = (now_us > peer.curr_recv_us) ? static_cast<double>(now_us - peer.curr_recv_us) : 0.0;
  const float t = static_cast<float>(std::clamp(elapsed / std::max(dt, 1000.0), 0.0, 1.0));
  SkaterState out = peer.curr_state;
  for (int i = 0; i < 3; ++i) {
    out.pos[i] = peer.prev_state.pos[i] + t * (peer.curr_state.pos[i] - peer.prev_state.pos[i]);
    out.vel[i] = peer.prev_state.vel[i] + t * (peer.curr_state.vel[i] - peer.prev_state.vel[i]);
  }
  // Gentle velocity dead-reckoning if a packet is slightly late (clamped to 50ms).
  if (elapsed > dt) {
    const float extra_s = static_cast<float>(std::min((elapsed - dt) * 1e-6, 0.05));
    for (int i = 0; i < 3; ++i) out.pos[i] += out.vel[i] * extra_s;
  }
  nlerp_quat(peer.prev_state.quat, peer.curr_state.quat, t, out.quat);
  nlerp_quat(peer.prev_state.board_quat, peer.curr_state.board_quat, t, out.board_quat);
  return out;
}

}  // namespace s3net
