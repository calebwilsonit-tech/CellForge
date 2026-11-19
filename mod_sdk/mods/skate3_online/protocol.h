#pragma once
// Skate3-VK Online (Path A: Direct P2P / UDP Relay) — Wire Protocol (v1)
// Compact, little-endian binary UDP packets for 60 Hz skater state sync.

#include <cstdint>
#include <cstring>

namespace s3net {

constexpr uint32_t kMagic = 0x4E4F3353u;  // "S3ON"
constexpr uint16_t kProtocolVersion = 1;
constexpr uint16_t kDefaultPort = 37464;  // BLUS30464 -> 37464
constexpr size_t kMaxPlayers = 8;
constexpr size_t kRoomCodeLen = 8;
constexpr size_t kNameLen = 24;

enum class MsgType : uint8_t {
  Hello = 1,      // Client -> Host/Relay: join room with player name
  Welcome = 2,    // Host/Relay -> Client: assigned peer_id + current peers
  State = 3,      // Bidirectional 60 Hz skater transform + animation state
  Event = 4,      // Reliable-ish gameplay event (session marker, trick, chat)
  Ping = 5,       // RTT probe
  Pong = 6,       // RTT reply
  Leave = 7,      // Graceful disconnect
};

#pragma pack(push, 1)

struct Header {
  uint32_t magic = kMagic;
  uint16_t version = kProtocolVersion;
  uint8_t type = 0;
  uint8_t peer_id = 0;      // 0 = unassigned / host
  uint32_t seq = 0;         // monotonic packet sequence
  uint32_t timestamp_ms = 0;
};

struct HelloPacket {
  Header hdr;
  char room[kRoomCodeLen]{};
  char name[kNameLen]{};
};

struct WelcomePacket {
  Header hdr;
  uint8_t assigned_peer_id = 0;
  uint8_t player_count = 0;
  uint16_t tick_hz = 60;
};

// 56 bytes of skater kinematics + state (fits easily in a single UDP datagram).
struct SkaterState {
  float pos[3]{};           // World XYZ
  float vel[3]{};           // Linear velocity XYZ
  float quat[4]{0, 0, 0, 1};// Orientation quaternion (x, y, z, w)
  float board_quat[4]{0, 0, 0, 1};
  uint32_t flags = 0;       // Bit0: on_board, Bit1: grinding, Bit2: airborne, Bit3: bailed
  uint32_t anim_id = 0;     // Current animation / trick hash
};

struct StatePacket {
  Header hdr;
  char room[kRoomCodeLen]{};
  SkaterState state;
};

struct EventPacket {
  Header hdr;
  char room[kRoomCodeLen]{};
  uint16_t event_kind = 0;  // 1=chat, 2=marker_teleport, 3=trick_callout
  uint16_t payload_len = 0;
  char payload[128]{};
};

struct PingPacket {
  Header hdr;
  uint64_t send_time_us = 0;
};

#pragma pack(pop)

static_assert(sizeof(Header) == 16, "Header must be 16 bytes");
static_assert(sizeof(SkaterState) == 64, "SkaterState must be 64 bytes");

inline void copy_str(char* dst, size_t cap, const char* src) {
  if (!dst || cap == 0) return;
  std::memset(dst, 0, cap);
  if (!src) return;
  const size_t n = std::strlen(src);
  std::memcpy(dst, src, n < cap - 1 ? n : cap - 1);
}

}  // namespace s3net
