// Sony controllers over HID: DualSense (PS5), DualSense Edge and DualShock 4
// (PS4), on USB or Bluetooth. Read with Windows' built-in HID API, so no driver or
// Steam Input / DS4Windows layer is needed.
//
// Input report layouts (public knowledge from the controllers' HID descriptors):
//   DualSense USB 0x01 / BT 0x31: LX LY RX RY L2 R2 seq B0 B1 B2 ...
//   DualShock 4 USB 0x01 / BT 0x11, and DualSense BT "simple" 0x01: LX LY RX RY B0 B1 B2 L2 R2
//   B0: hat (low nibble: 0 = N .. 7 = NW, 8 = none), Square 0x10, Cross 0x20, Circle 0x40, Triangle 0x80
//   B1: L1 0x01, R1 0x02, L2 0x04, R2 0x08, Create/Share 0x10, Options 0x20, L3 0x40, R3 0x80
//   B2: PS 0x01, touchpad click 0x02

#include "input.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <hidsdi.h>
#include <setupapi.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "hid.lib")
#pragma comment(lib, "setupapi.lib")

namespace input {
namespace {

constexpr USHORT kSony = 0x054C;
bool is_dualsense(USHORT pid) { return pid == 0x0CE6 || pid == 0x0DF2; }
bool is_ds4(USHORT pid) { return pid == 0x05C4 || pid == 0x09CC || pid == 0x0BA0; }

struct Pad {
  std::wstring path;
  HANDLE h = INVALID_HANDLE_VALUE;
  bool dualsense = false;
  std::mutex m;
  PadState state;
  std::atomic<bool> alive{true};
};

std::mutex g_m;
std::vector<std::shared_ptr<Pad>> g_pads;  // index = controller number

void decode(const uint8_t* p, bool ds5_layout, PadState& s) {
  // p points at LX.
  const uint8_t b0 = ds5_layout ? p[7] : p[4];
  const uint8_t b1 = ds5_layout ? p[8] : p[5];
  const uint8_t b2 = ds5_layout ? p[9] : p[6];
  const uint8_t l2 = ds5_layout ? p[4] : p[7];
  const uint8_t r2 = ds5_layout ? p[5] : p[8];
  PadState n;
  n.connected = true;
  n.lx = p[0]; n.ly = p[1]; n.rx = p[2]; n.ry = p[3];
  switch (b0 & 0x0F) {
    case 0: n.digital1 |= UP; break;
    case 1: n.digital1 |= UP | RIGHT; break;
    case 2: n.digital1 |= RIGHT; break;
    case 3: n.digital1 |= DOWN | RIGHT; break;
    case 4: n.digital1 |= DOWN; break;
    case 5: n.digital1 |= DOWN | LEFT; break;
    case 6: n.digital1 |= LEFT; break;
    case 7: n.digital1 |= UP | LEFT; break;
    default: break;
  }
  if (b0 & 0x10) n.digital2 |= SQUARE;
  if (b0 & 0x20) n.digital2 |= CROSS;
  if (b0 & 0x40) n.digital2 |= CIRCLE;
  if (b0 & 0x80) n.digital2 |= TRIANGLE;
  if (b1 & 0x01) n.digital2 |= L1;
  if (b1 & 0x02) n.digital2 |= R1;
  if (b1 & 0x04) n.digital2 |= L2;
  if (b1 & 0x08) n.digital2 |= R2;
  if (b1 & 0x10) n.digital1 |= SELECT;  // Create / Share
  if (b1 & 0x20) n.digital1 |= START;   // Options
  if (b1 & 0x40) n.digital1 |= L3;
  if (b1 & 0x80) n.digital1 |= R3;
  if (b2 & 0x02) n.digital1 |= SELECT;  // touchpad click doubles as Select
  n.pressure[10] = l2;
  n.pressure[11] = r2;
  s = n;
}

void reader(std::shared_ptr<Pad> pad) {
  std::vector<uint8_t> buf(128);
  OVERLAPPED ov{};
  ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
  while (pad->alive) {
    DWORD got = 0;
    ResetEvent(ov.hEvent);
    if (!ReadFile(pad->h, buf.data(), DWORD(buf.size()), &got, &ov)) {
      if (GetLastError() != ERROR_IO_PENDING) break;
      if (WaitForSingleObject(ov.hEvent, 1000) != WAIT_OBJECT_0) { CancelIo(pad->h); continue; }
      if (!GetOverlappedResult(pad->h, &ov, &got, FALSE)) break;
    }
    if (got < 10) continue;
    PadState s;
    const uint8_t id = buf[0];
    if (pad->dualsense && id == 0x01 && got >= 11 + 1 && got > 20) decode(buf.data() + 1, true, s);  // USB
    else if (pad->dualsense && id == 0x31 && got >= 12) decode(buf.data() + 2, true, s);              // BT extended
    else if (id == 0x11 && got >= 12) decode(buf.data() + 3, false, s);                               // DS4 BT
    else if (id == 0x01) decode(buf.data() + 1, false, s);                                            // DS4 USB / BT simple
    else continue;
    std::lock_guard<std::mutex> lk(pad->m);
    pad->state = s;
  }
  CloseHandle(ov.hEvent);
  pad->alive = false;
  std::lock_guard<std::mutex> lk(pad->m);
  pad->state = PadState{};
}

// Finds Sony controllers that are not open yet.
void scan() {
  GUID hid;
  HidD_GetHidGuid(&hid);
  HDEVINFO set = SetupDiGetClassDevsW(&hid, nullptr, nullptr, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
  if (set == INVALID_HANDLE_VALUE) return;
  SP_DEVICE_INTERFACE_DATA ifd{sizeof ifd};
  for (DWORD i = 0; SetupDiEnumDeviceInterfaces(set, nullptr, &hid, i, &ifd); ++i) {
    DWORD need = 0;
    SetupDiGetDeviceInterfaceDetailW(set, &ifd, nullptr, 0, &need, nullptr);
    std::vector<uint8_t> mem(need);
    auto* detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(mem.data());
    detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
    if (!SetupDiGetDeviceInterfaceDetailW(set, &ifd, detail, need, nullptr, nullptr)) continue;
    const std::wstring path = detail->DevicePath;
    {
      std::lock_guard<std::mutex> lk(g_m);
      bool open = false;
      for (auto& p : g_pads) open |= p->alive && p->path == path;
      if (open) continue;
    }
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) continue;
    HIDD_ATTRIBUTES attr{sizeof attr};
    if (!HidD_GetAttributes(h, &attr) || attr.VendorID != kSony || !(is_dualsense(attr.ProductID) || is_ds4(attr.ProductID))) {
      CloseHandle(h);
      continue;
    }
    // A DualSense on Bluetooth sends the full report (0x31) once feature report 0x05 has been read.
    if (is_dualsense(attr.ProductID)) {
      uint8_t feature[64] = {0x05};
      HidD_GetFeature(h, feature, sizeof feature);
    }
    auto pad = std::make_shared<Pad>();
    pad->path = path;
    pad->h = h;
    pad->dualsense = is_dualsense(attr.ProductID);
    {
      std::lock_guard<std::mutex> lk(g_m);
      // Reuse a dead slot so controller numbers stay stable.
      bool placed = false;
      for (auto& p : g_pads)
        if (!p->alive) { p = pad; placed = true; break; }
      if (!placed) g_pads.push_back(pad);
    }
    std::thread(reader, pad).detach();
  }
  SetupDiDestroyDeviceInfoList(set);
}

void start_scanner() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::thread([] {
      for (;;) {
        scan();
        std::this_thread::sleep_for(std::chrono::seconds(2));  // picks up controllers plugged in later
      }
    }).detach();
  });
}

}  // namespace

bool sony_pad_read(int index, PadState& out) {
  start_scanner();
  std::shared_ptr<Pad> pad;
  {
    std::lock_guard<std::mutex> lk(g_m);
    int n = 0;
    for (auto& p : g_pads) {
      if (!p->alive) continue;
      if (n++ == index) { pad = p; break; }
    }
  }
  if (!pad) return false;
  std::lock_guard<std::mutex> lk(pad->m);
  if (!pad->state.connected) return false;
  out = pad->state;
  return true;
}

}  // namespace input

#else
namespace input {
bool sony_pad_read(int, PadState&) { return false; }
}  // namespace input
#endif
