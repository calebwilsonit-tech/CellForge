#pragma once
// Host controller input for cellPad. A provider fills a PadState per port; the
// default provider on Windows reads XInput controllers (Xbox-style pads).

#include <cstdint>
#include <functional>

namespace input {

// Button bits, matching CellPad's DIGITAL1 / DIGITAL2 words.
enum Digital1 : uint16_t {
  SELECT = 0x01, L3 = 0x02, R3 = 0x04, START = 0x08, UP = 0x10, RIGHT = 0x20, DOWN = 0x40, LEFT = 0x80
};
enum Digital2 : uint16_t {
  L2 = 0x01, R2 = 0x02, L1 = 0x04, R1 = 0x08, TRIANGLE = 0x10, CIRCLE = 0x20, CROSS = 0x40, SQUARE = 0x80
};

struct PadState {
  bool connected = false;
  uint16_t digital1 = 0, digital2 = 0;
  uint8_t lx = 0x80, ly = 0x80, rx = 0x80, ry = 0x80;  // 0..255, 0x80 = centre, up = 0
  // Pressure (0..255): right, left, up, down, triangle, circle, cross, square, L1, R1, L2, R2
  uint8_t pressure[12] = {};
  uint16_t sensor_x = 512, sensor_y = 399, sensor_z = 512, sensor_g = 512;
};

constexpr int kMaxPads = 7;

using Provider = std::function<bool(int port, PadState& out)>;

// Replace the input source (a window's keyboard handler, a test, or a mod).
void set_provider(Provider p);
// Reads one port through the current provider. Pressure values that the provider
// left at zero are derived from the digital buttons.
PadState read(int port);

// HID controllers (DualSense / DualShock 4 / generic USB & Bluetooth HID gamepads).
bool sony_pad_read(int index, PadState& out);
int sony_pad_count();

}  // namespace input
