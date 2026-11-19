// Skate3-VK Online Mod (Path A: Direct P2P / UDP Relay)
// Drop `skate3_online.dll` and `skate3_online.ini` into your game's `mods/` folder.
//
// Features:
// - Native Windows Winsock2 UDP networking (no PS3 sys_net required!)
// - Works over LAN, Tailscale/ZeroTier, port-forwarding, or the bundled `relay_server.py`
// - 60 Hz skater transform sync + client-side interpolation
// - Configurable BLUS30464 guest memory bridge for local skater & remote ghost slots

#include "cellforge_mod.h"
#include "net_client.h"

#include <chrono>
#include <cstdlib>
#include <string>

namespace {

s3net::NetClient g_net;
cellforge::Hook g_frame_hook;

// Configurable guest memory addresses (set in mods/skate3_online.ini once mapped for your EBOOT)
uint32_t g_skater_ptr_addr = 0;     // Guest pointer to local skater transform struct (pos[3], vel[3], quat[4])
uint32_t g_ghost_pool_addr = 0;     // Guest base address of remote ghost/NPC transform array
uint32_t g_ghost_stride = 0x40;     // Byte stride between remote ghost slots

uint64_t steady_us() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void sync_skater_tick() {
  // 1. Read local skater state from PS3 guest RAM (big-endian floats) if configured.
  if (g_skater_ptr_addr != 0) {
    const uint32_t entity = cellforge::read_u32(g_skater_ptr_addr);
    if (entity >= 0x00100000u && entity < 0x50000000u) {
      s3net::SkaterState st{};
      for (int i = 0; i < 3; ++i) st.pos[i] = cellforge::read_f32(entity + i * 4);
      for (int i = 0; i < 3; ++i) st.vel[i] = cellforge::read_f32(entity + 12 + i * 4);
      for (int i = 0; i < 4; ++i) st.quat[i] = cellforge::read_f32(entity + 24 + i * 4);
      g_net.set_local_state(st);
    }
  }

  // 2. Write interpolated remote peers into ghost/NPC slots if configured.
  if (g_ghost_pool_addr != 0) {
    const uint64_t now = steady_us();
    const auto peers = g_net.get_remote_peers();
    uint32_t slot = 0;
    for (const auto& p : peers) {
      if (slot >= s3net::kMaxPlayers - 1) break;
      const s3net::SkaterState interp = s3net::NetClient::interpolate(p, now);
      const uint32_t dst = g_ghost_pool_addr + slot * g_ghost_stride;
      for (int i = 0; i < 3; ++i) cellforge::write_f32(dst + i * 4, interp.pos[i]);
      for (int i = 0; i < 3; ++i) cellforge::write_f32(dst + 12 + i * 4, interp.vel[i]);
      for (int i = 0; i < 4; ++i) cellforge::write_f32(dst + 24 + i * 4, interp.quat[i]);
      ++slot;
    }
  }
}

void hooked_frame_update(ppu::Context& c) {
  g_frame_hook.call_original(c);
  sync_skater_tick();
}

uint32_t parse_hex(const std::string& s) {
  if (s.empty()) return 0;
  return static_cast<uint32_t>(std::strtoul(s.c_str(), nullptr, 16));
}

}  // namespace

CELLFORGE_MOD_ENTRY(api) {
  if (!cellforge::init(api)) return 1;

  cellforge::Config cfg;
  const bool loaded = cfg.load("mods/skate3_online.ini");
  if (!cfg.get_bool("enabled", true)) {
    cellforge::log("skate3_online: disabled in mods/skate3_online.ini");
    return 0;
  }

  s3net::ClientConfig net_cfg;
  net_cfg.mode = cfg.get_string("mode", "relay");
  net_cfg.host = cfg.get_string("host", "127.0.0.1");
  net_cfg.port = static_cast<uint16_t>(cfg.get_int("port", s3net::kDefaultPort));
  net_cfg.room = cfg.get_string("room", "SK8A");
  net_cfg.player_name = cfg.get_string("player_name", "Skater");
  net_cfg.tick_hz = static_cast<uint16_t>(cfg.get_int("tick_hz", 60));

  g_skater_ptr_addr = parse_hex(cfg.get_string("skater_ptr_addr", ""));
  g_ghost_pool_addr = parse_hex(cfg.get_string("ghost_pool_addr", ""));
  const uint32_t frame_hook_addr = parse_hex(cfg.get_string("frame_hook_addr", ""));

  cellforge::logf("skate3_online: starting (config=%s, mode=%s, host=%s:%u, room=%s)",
                  loaded ? "mods/skate3_online.ini" : "defaults",
                  net_cfg.mode.c_str(), net_cfg.host.c_str(), unsigned(net_cfg.port), net_cfg.room.c_str());

  if (!g_net.start(net_cfg, [](const std::string& m) { cellforge::logf("skate3_online: %s", m.c_str()); })) {
    cellforge::log("skate3_online: failed to start UDP transport");
    return 1;
  }

  if (frame_hook_addr != 0) {
    if (g_frame_hook.install(frame_hook_addr, hooked_frame_update)) {
      cellforge::logf("skate3_online: hooked frame tick at 0x%08X", frame_hook_addr);
    } else {
      cellforge::logf("skate3_online: warning: frame_hook_addr 0x%08X is not a recompiled function", frame_hook_addr);
    }
  } else {
    cellforge::log("skate3_online: transport active (configure skater_ptr_addr & frame_hook_addr in skate3_online.ini for live entity sync)");
  }

  return 0;
}
