#define _CRT_SECURE_NO_WARNINGS
#include <atomic>
#include "input.h"
#include "rsx.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace input {
namespace {

#if defined(_WIN32)
// XInput, loaded at runtime so the program still starts without it.
struct XGamepad {
  WORD buttons;
  BYTE lt, rt;
  SHORT lx, ly, rx, ry;
};
struct XState {
  DWORD packet;
  XGamepad pad;
};
using XInputGetStateFn = DWORD(WINAPI*)(DWORD, XState*);

XInputGetStateFn xinput() {
  static XInputGetStateFn fn = [] {
    for (const wchar_t* dll : {L"xinput1_4.dll", L"xinput1_3.dll", L"xinput9_1_0.dll"})
      if (HMODULE m = LoadLibraryW(dll))
        if (auto f = reinterpret_cast<XInputGetStateFn>(reinterpret_cast<void*>(GetProcAddress(m, "XInputGetState"))))
          return f;
    return XInputGetStateFn(nullptr);
  }();
  return fn;
}

uint8_t axis(SHORT v, bool invert) {
  const int u = (int(v) + 32768) >> 8;  // 0..255
  return uint8_t(invert ? 255 - u : u);
}

bool xinput_provider(int port, PadState& s) {
  // HID controllers (DualSense, DualShock 4, Switch Pro, 8BitDo, generic USB/BT gamepads)
  // take the first ports; XInput controllers (Xbox USB/BT) fill in after them.
  if (sony_pad_read(port, s)) return true;
  auto get = xinput();
  if (!get) return false;
  const int xi_target = port - sony_pad_count();
  if (xi_target < 0 || xi_target >= 4) return false;
  // Scan all 4 XInput slots and compact connected pads so a Bluetooth Xbox controller
  // that reconnects on slot 1/2/3 still maps to Player 1 when slot 0 is empty.
  XState st{};
  int found = 0;
  bool ok = false;
  for (DWORD slot = 0; slot < 4; ++slot) {
    if (get(slot, &st) == 0) {
      if (found == xi_target) { ok = true; break; }
      ++found;
    }
  }
  if (!ok) return false;
  const WORD b = st.pad.buttons;
  s.connected = true;
  if (b & 0x0001) s.digital1 |= UP;
  if (b & 0x0002) s.digital1 |= DOWN;
  if (b & 0x0004) s.digital1 |= LEFT;
  if (b & 0x0008) s.digital1 |= RIGHT;
  if (b & 0x0010) s.digital1 |= START;
  if (b & 0x0020) s.digital1 |= SELECT;
  if (b & 0x0040) s.digital1 |= L3;
  if (b & 0x0080) s.digital1 |= R3;
  if (b & 0x0100) s.digital2 |= L1;
  if (b & 0x0200) s.digital2 |= R1;
  if (b & 0x1000) s.digital2 |= CROSS;     // A
  if (b & 0x2000) s.digital2 |= CIRCLE;    // B
  if (b & 0x4000) s.digital2 |= SQUARE;    // X
  if (b & 0x8000) s.digital2 |= TRIANGLE;  // Y
  if (st.pad.lt > 30) s.digital2 |= L2;
  if (st.pad.rt > 30) s.digital2 |= R2;
  s.pressure[10] = st.pad.lt;
  s.pressure[11] = st.pad.rt;
  s.lx = axis(st.pad.lx, false);
  s.ly = axis(st.pad.ly, true);
  s.rx = axis(st.pad.rx, false);
  s.ry = axis(st.pad.ry, true);
  return true;
}
#endif

std::mutex g_m;
Provider g_provider =
#if defined(_WIN32)
    xinput_provider;
#else
    nullptr;
#endif

// PS3RECOMP_INPUT_SCRIPT="start@60,cross@f1200,...": presses a button on pad 0 at
// the given seconds after the first pad read (held for 250 ms), or at RSX flip N
// when prefixed with 'f' (held for 8 flips). For automated test runs without a
// controller; the scripted pad counts as connected.
struct ScriptedPress { double at; bool by_flip; int word; uint16_t bit; };
const std::vector<ScriptedPress>& input_script() {
  static const std::vector<ScriptedPress> script = [] {
    std::vector<ScriptedPress> out;
    const char* env = std::getenv("PS3RECOMP_INPUT_SCRIPT");
    if (!env) return out;
    const struct { const char* name; int word; uint16_t bit; } names[] = {
        {"select", 1, SELECT}, {"l3", 1, L3}, {"r3", 1, R3}, {"start", 1, START}, {"up", 1, UP}, {"right", 1, RIGHT},
        {"down", 1, DOWN}, {"left", 1, LEFT}, {"l2", 2, L2}, {"r2", 2, R2}, {"l1", 2, L1}, {"r1", 2, R1},
        {"triangle", 2, TRIANGLE}, {"circle", 2, CIRCLE}, {"cross", 2, CROSS}, {"square", 2, SQUARE}};
    std::string s(env);
    size_t p = 0;
    while (p < s.size()) {
      const size_t comma = s.find(',', p);
      const std::string item = s.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
      const size_t at = item.find('@');
      if (at != std::string::npos) {
        const char* val = item.c_str() + at + 1;
        const bool by_flip = (*val == 'f' || *val == 'F');
        if (by_flip) ++val;
        for (const auto& n : names)
          if (item.compare(0, at, n.name) == 0 && std::strlen(n.name) == at)
            out.push_back({std::atof(val), by_flip, n.word, n.bit});
      }
      if (comma == std::string::npos) break;
      p = comma + 1;
    }
    return out;
  }();
  return script;
}

}  // namespace

void set_provider(Provider p) {
  std::lock_guard<std::mutex> lk(g_m);
  g_provider = std::move(p);
}

static std::atomic<bool> g_game_blocked{false};
void set_game_blocked(bool b) { g_game_blocked = b; }
bool game_blocked() { return g_game_blocked.load(); }

PadState read(int port) {
  Provider p;
  {
    std::lock_guard<std::mutex> lk(g_m);
    p = g_provider;
  }
  PadState s;
  const bool real = p && p(port, s);
  const auto& script = input_script();
  if (port == 0 && !script.empty()) {
    static const auto t0 = std::chrono::steady_clock::now();
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t flips = rsx::Rsx::get().stats.flips.load(std::memory_order_relaxed);
    s.connected = true;
    for (const auto& e : script) {
      const bool active = e.by_flip ? (flips >= uint64_t(e.at) && flips < uint64_t(e.at) + 8)
                                    : (now >= e.at && now < e.at + 0.25);
      if (active) (e.word == 1 ? s.digital1 : s.digital2) |= e.bit;
    }
  } else if (!real) {
    return PadState{};
  }
  // Derive pressure from digital buttons where the provider gave none.
  const struct { int idx; uint16_t word; uint16_t bit; } map[] = {
      {0, 1, RIGHT}, {1, 1, LEFT}, {2, 1, UP}, {3, 1, DOWN}, {4, 2, TRIANGLE}, {5, 2, CIRCLE},
      {6, 2, CROSS}, {7, 2, SQUARE}, {8, 2, L1}, {9, 2, R1}, {10, 2, L2}, {11, 2, R2}};
  for (const auto& m : map) {
    const bool down = ((m.word == 1 ? s.digital1 : s.digital2) & m.bit) != 0;
    if (down && s.pressure[m.idx] == 0) s.pressure[m.idx] = 255;
  }
  return s;
}

}  // namespace input
