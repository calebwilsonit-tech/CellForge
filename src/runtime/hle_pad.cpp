// cellPad (sys_io) HLE, fed by input::read().

#include <cstring>
#include <mutex>

#include "hle.h"
#include "input.h"
#include "lv2.h"

namespace hle {
namespace {

using lv2::Kernel;

// Error codes from the public cellPad documentation.
constexpr uint32_t CELL_PAD_OK = 0;
constexpr uint32_t CELL_PAD_ERROR_ALREADY_INITIALIZED = 0x80121101;
constexpr uint32_t CELL_PAD_ERROR_UNINITIALIZED = 0x80121103;
constexpr uint32_t CELL_PAD_ERROR_INVALID_PARAMETER = 0x80121104;
constexpr uint32_t CELL_PAD_ERROR_NO_DEVICE = 0x80121107;

constexpr uint32_t STATUS_CONNECTED = 1, STATUS_ASSIGN_CHANGES = 2;
// PS3 conformity | press mode | sensor mode | high-precision sticks | actuator
constexpr uint32_t kCapability = 0x1F;
constexpr uint32_t kPadDataLen = 24;  // number of valid u16 button words

Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }

std::mutex g_m;
bool g_init = false;
uint32_t g_max = 0;
uint32_t g_setting[input::kMaxPads] = {};
bool g_was_connected[input::kMaxPads] = {};

bool ready(ppu::Context& c) {
  if (!g_init) { ret(c, CELL_PAD_ERROR_UNINITIALIZED); return false; }
  return true;
}

void cellPadInit(ppu::Context& c) {
  std::lock_guard<std::mutex> lk(g_m);
  if (g_init) return ret(c, CELL_PAD_ERROR_ALREADY_INITIALIZED);
  const uint32_t max = a32(c, 0);
  if (max == 0 || max > uint32_t(input::kMaxPads)) return ret(c, CELL_PAD_ERROR_INVALID_PARAMETER);
  g_init = true;
  g_max = max;
  ret(c, CELL_PAD_OK);
}
void cellPadEnd(ppu::Context& c) {
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  g_init = false;
  ret(c, CELL_PAD_OK);
}

void cellPadGetData(ppu::Context& c) {
  // (u32 port, CellPadData* data): s32 len, u16 button[64]
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t port = a32(c, 0), out = a32(c, 1);
  if (port >= g_max || !out) return ret(c, CELL_PAD_ERROR_INVALID_PARAMETER);
  input::PadState s = input::read(int(port));
  if (!s.connected) return ret(c, CELL_PAD_ERROR_NO_DEVICE);
  if (input::game_blocked()) {
    const input::PadState idle;
    s = idle;
    s.connected = true;
  }

  uint16_t b[24] = {};
  b[0] = 0;
  b[1] = uint16_t(0x0070 | (kPadDataLen / 2));
  b[2] = s.digital1;
  b[3] = s.digital2;
  b[4] = s.rx; b[5] = s.ry; b[6] = s.lx; b[7] = s.ly;
  for (int i = 0; i < 12; ++i) b[8 + i] = s.pressure[i];
  b[20] = s.sensor_x; b[21] = s.sensor_y; b[22] = s.sensor_z; b[23] = s.sensor_g;
  K().w32(out, kPadDataLen);
  for (int i = 0; i < 24; ++i) ppu::st16(K().base(), out + 4 + 2 * i, b[i]);
  ret(c, CELL_PAD_OK);
}

void cellPadGetInfo2(ppu::Context& c) {
  // CellPadInfo2: u32 max_connect, now_connect, system_info,
  //               port_status[7], port_setting[7], device_capability[7], device_type[7]
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t a = a32(c, 0);
  uint32_t now = 0;
  for (int p = 0; p < input::kMaxPads; ++p) {
    const bool conn = uint32_t(p) < g_max && input::read(p).connected;
    uint32_t status = conn ? STATUS_CONNECTED : 0;
    if (conn != g_was_connected[p]) status |= STATUS_ASSIGN_CHANGES;
    g_was_connected[p] = conn;
    now += conn;
    K().w32(a + 12 + 4 * p, status);
    K().w32(a + 40 + 4 * p, g_setting[p]);
    K().w32(a + 68 + 4 * p, conn ? kCapability : 0);
    K().w32(a + 96 + 4 * p, 0);  // standard controller
  }
  K().w32(a + 0, g_max);
  K().w32(a + 4, now);
  K().w32(a + 8, 0);
  ret(c, CELL_PAD_OK);
}

void cellPadGetInfo(ppu::Context& c) {
  // CellPadInfo: u32 max_connect, now_connect, system_info, u16 vendor_id[127],
  //              u16 product_id[127], u8 status[127]
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t a = a32(c, 0);
  uint32_t now = 0;
  for (int p = 0; p < 127; ++p) {
    const bool conn = p < input::kMaxPads && uint32_t(p) < g_max && input::read(p).connected;
    now += conn;
    ppu::st16(K().base(), a + 12 + 2 * p, conn ? 0x054C : 0);          // Sony
    ppu::st16(K().base(), a + 12 + 254 + 2 * p, conn ? 0x0268 : 0);    // DualShock 3
    K().base()[a + 12 + 508 + p] = conn ? STATUS_CONNECTED : 0;
  }
  K().w32(a + 0, g_max);
  K().w32(a + 4, now);
  K().w32(a + 8, 0);
  ret(c, CELL_PAD_OK);
}

void cellPadSetPortSetting(ppu::Context& c) {
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t port = a32(c, 0);
  if (port >= uint32_t(input::kMaxPads)) return ret(c, CELL_PAD_ERROR_INVALID_PARAMETER);
  g_setting[port] = a32(c, 1);
  ret(c, CELL_PAD_OK);
}
void set_mode_bit(ppu::Context& c, uint32_t bit) {
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t port = a32(c, 0);
  if (port >= uint32_t(input::kMaxPads)) return ret(c, CELL_PAD_ERROR_INVALID_PARAMETER);
  if (a32(c, 1)) g_setting[port] |= bit; else g_setting[port] &= ~bit;
  ret(c, CELL_PAD_OK);
}
void info_mode_bit(ppu::Context& c, uint32_t bit) {
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t port = a32(c, 0);
  if (port >= uint32_t(input::kMaxPads)) return ret(c, CELL_PAD_ERROR_INVALID_PARAMETER);
  ret(c, (g_setting[port] & bit) ? 1 : 0);
}
void cellPadSetPressMode(ppu::Context& c) { set_mode_bit(c, 2); }
void cellPadSetSensorMode(ppu::Context& c) { set_mode_bit(c, 4); }
void cellPadInfoPressMode(ppu::Context& c) { info_mode_bit(c, 2); }
void cellPadInfoSensorMode(ppu::Context& c) { info_mode_bit(c, 4); }

void cellPadGetCapabilityInfo(ppu::Context& c) {
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  const uint32_t a = a32(c, 1);
  std::memset(K().base() + a, 0, 128);
  K().w32(a, kCapability);
  ret(c, CELL_PAD_OK);
}

void pad_ok(ppu::Context& c) {
  std::lock_guard<std::mutex> lk(g_m);
  if (!ready(c)) return;
  ret(c, CELL_PAD_OK);
}

}  // namespace

void register_pad_modules() {
#define R(fn) register_function("sys_io", #fn, fn)
  R(cellPadInit); R(cellPadEnd); R(cellPadGetData); R(cellPadGetInfo); R(cellPadGetInfo2);
  R(cellPadSetPortSetting); R(cellPadSetPressMode); R(cellPadSetSensorMode); R(cellPadInfoPressMode);
  R(cellPadInfoSensorMode); R(cellPadGetCapabilityInfo);
#undef R
  register_function("sys_io", "cellPadClearBuf", pad_ok);
  register_function("sys_io", "cellPadSetActDirect", pad_ok);
}

}  // namespace hle
