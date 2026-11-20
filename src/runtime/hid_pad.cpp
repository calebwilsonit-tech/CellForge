// Universal Windows HID controller backend:
// - Native fast-path for Sony DualSense / DualSense Edge / DualShock 4 (USB & Bluetooth)
// - Generic HID descriptor-driven parser for ANY other USB or Bluetooth gamepad/joystick
//   (Nintendo Switch Pro, 8BitDo, Logitech, Stadia, third-party HID/DirectInput pads).
// XInput devices (HID path containing "IG_") are skipped here and handled by XInput in input.cpp.

#include "input.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <hidsdi.h>
#include <hidpi.h>
#include <setupapi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cwctype>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "hid.lib")
#pragma comment(lib, "setupapi.lib")

namespace input {
namespace {

constexpr USHORT kSony = 0x054C;
constexpr USHORT kNintendo = 0x057E;
bool is_dualsense(USHORT pid) { return pid == 0x0CE6 || pid == 0x0DF2; }
bool is_ds4(USHORT pid) { return pid == 0x05C4 || pid == 0x09CC || pid == 0x0BA0; }

bool path_contains_ig(const std::wstring& path) {
  std::wstring up = path;
  for (wchar_t& ch : up) ch = std::towupper(ch);
  return up.find(L"IG_") != std::wstring::npos;
}

struct AxisInfo {
  bool present = false;
  USAGE page = 0;
  USAGE usage = 0;
  LONG logical_min = 0;
  LONG logical_max = 255;
  USHORT bit_size = 8;
};

struct Pad {
  std::wstring path;
  HANDLE h = INVALID_HANDLE_VALUE;
  USHORT vid = 0, pid = 0;
  bool sony = false;
  bool dualsense = false;
  bool is_bt = false;
  USHORT input_report_len = 64;

  // Generic HID descriptor info (when !sony)
  PHIDP_PREPARSED_DATA ppd = nullptr;
  USHORT num_buttons = 0;
  AxisInfo ax_x, ax_y, ax_z, ax_rx, ax_ry, ax_rz, ax_hat, ax_brake, ax_accel;

  std::mutex m;
  PadState state;
  std::atomic<bool> alive{true};

  ~Pad() {
    if (ppd) HidD_FreePreparsedData(ppd);
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
  }
};

std::mutex g_m;
std::vector<std::shared_ptr<Pad>> g_pads;  // index = controller number

void decode_sony(const uint8_t* p, bool ds5_layout, PadState& s) {
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

uint8_t read_axis_u8(const Pad& pad, const AxisInfo& ax, const uint8_t* report, ULONG len, uint8_t def = 0x80) {
  if (!ax.present || !pad.ppd) return def;
  LONG val = 0;
  if (ax.logical_min < 0) {
    if (HidP_GetScaledUsageValue(HidP_Input, ax.page, 0, ax.usage, &val, pad.ppd,
                                 reinterpret_cast<PCHAR>(const_cast<uint8_t*>(report)), len) != HIDP_STATUS_SUCCESS)
      return def;
  } else {
    ULONG uval = 0;
    if (HidP_GetUsageValue(HidP_Input, ax.page, 0, ax.usage, &uval, pad.ppd,
                           reinterpret_cast<PCHAR>(const_cast<uint8_t*>(report)), len) != HIDP_STATUS_SUCCESS)
      return def;
    val = LONG(uval);
  }
  const int64_t lo = ax.logical_min;
  const int64_t hi = (ax.logical_max > ax.logical_min) ? ax.logical_max : ((1LL << std::min<int>(ax.bit_size, 16)) - 1);
  if (hi <= lo) return def;
  const int64_t clamped = std::clamp<int64_t>(val, lo, hi);
  return uint8_t(((clamped - lo) * 255) / (hi - lo));
}

bool decode_generic(const Pad& pad, const uint8_t* report, ULONG len, PadState& s) {
  if (!pad.ppd) return false;
  PadState n;
  n.connected = true;

  // Sticks: LX=X, LY=Y; RX/RY come from (Z, Rz) on PS/DInput/Switch pads or (Rx, Ry) on others.
  n.lx = read_axis_u8(pad, pad.ax_x, report, len, 0x80);
  n.ly = read_axis_u8(pad, pad.ax_y, report, len, 0x80);
  if (pad.ax_z.present && pad.ax_rz.present) {
    n.rx = read_axis_u8(pad, pad.ax_z, report, len, 0x80);
    n.ry = read_axis_u8(pad, pad.ax_rz, report, len, 0x80);
  } else if (pad.ax_rx.present && pad.ax_ry.present) {
    n.rx = read_axis_u8(pad, pad.ax_rx, report, len, 0x80);
    n.ry = read_axis_u8(pad, pad.ax_ry, report, len, 0x80);
  }

  // Optional analog triggers (Brake/Accelerator or Rx/Ry when Z/Rz are right stick).
  uint8_t l2 = 0, r2 = 0;
  if (pad.ax_brake.present) l2 = read_axis_u8(pad, pad.ax_brake, report, len, 0);
  if (pad.ax_accel.present) r2 = read_axis_u8(pad, pad.ax_accel, report, len, 0);
  if (l2 > 30) n.digital2 |= L2;
  if (r2 > 30) n.digital2 |= R2;
  n.pressure[10] = l2;
  n.pressure[11] = r2;

  // Hat switch (D-Pad): normalize against logical_min (usually 0..7 or 1..8).
  if (pad.ax_hat.present) {
    ULONG hv = 0;
    if (HidP_GetUsageValue(HidP_Input, pad.ax_hat.page, 0, pad.ax_hat.usage, &hv, pad.ppd,
                           reinterpret_cast<PCHAR>(const_cast<uint8_t*>(report)), len) == HIDP_STATUS_SUCCESS) {
      const LONG dir = LONG(hv) - pad.ax_hat.logical_min;
      const LONG span = pad.ax_hat.logical_max - pad.ax_hat.logical_min;
      if (span == 7) {
        switch (dir) {
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
      } else if (span == 3) {
        switch (dir) {
          case 0: n.digital1 |= UP; break;
          case 1: n.digital1 |= RIGHT; break;
          case 2: n.digital1 |= DOWN; break;
          case 3: n.digital1 |= LEFT; break;
          default: break;
        }
      }
    }
  }

  // Buttons (Usage Page 0x09, 1-based).
  USAGE usages[32]{};
  ULONG count = 32;
  if (HidP_GetUsages(HidP_Input, HID_USAGE_PAGE_BUTTON, 0, usages, &count, pad.ppd,
                     reinterpret_cast<PCHAR>(const_cast<uint8_t*>(report)), len) == HIDP_STATUS_SUCCESS) {
    const bool nintendo = (pad.vid == kNintendo);
    for (ULONG i = 0; i < count; ++i) {
      const USAGE b = usages[i];
      if (nintendo) {
        // Switch Pro HID button numbering: 1=B, 2=A, 3=Y, 4=X, 5=L, 6=R, 7=ZL, 8=ZR, 9=-, 10=+, 11=L3, 12=R3
        switch (b) {
          case 1: n.digital2 |= CROSS; break;
          case 2: n.digital2 |= CIRCLE; break;
          case 3: n.digital2 |= SQUARE; break;
          case 4: n.digital2 |= TRIANGLE; break;
          case 5: n.digital2 |= L1; break;
          case 6: n.digital2 |= R1; break;
          case 7: n.digital2 |= L2; break;
          case 8: n.digital2 |= R2; break;
          case 9: n.digital1 |= SELECT; break;
          case 10: n.digital1 |= START; break;
          case 11: n.digital1 |= L3; break;
          case 12: n.digital1 |= R3; break;
          default: break;
        }
      } else {
        // Standard HID / DirectInput gamepad numbering:
        // 1=Square/X, 2=Cross/A, 3=Circle/B, 4=Triangle/Y, 5=L1, 6=R1, 7=L2, 8=R2, 9=Select, 10=Start, 11=L3, 12=R3
        switch (b) {
          case 1: n.digital2 |= SQUARE; break;
          case 2: n.digital2 |= CROSS; break;
          case 3: n.digital2 |= CIRCLE; break;
          case 4: n.digital2 |= TRIANGLE; break;
          case 5: n.digital2 |= L1; break;
          case 6: n.digital2 |= R1; break;
          case 7: n.digital2 |= L2; break;
          case 8: n.digital2 |= R2; break;
          case 9: n.digital1 |= SELECT; break;
          case 10: n.digital1 |= START; break;
          case 11: n.digital1 |= L3; break;
          case 12: n.digital1 |= R3; break;
          default: break;
        }
      }
    }
  }

  s = n;
  return true;
}

void reader(std::shared_ptr<Pad> pad) {
  std::vector<uint8_t> buf(std::max<size_t>(128, pad->input_report_len));
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
    if (got < 2) continue;
    PadState s;
    if (pad->sony) {
      if (got < 10) continue;
      const uint8_t id = buf[0];
      if (pad->dualsense && !pad->is_bt && id == 0x01 && got >= 12) decode_sony(buf.data() + 1, true, s);       // DS5 USB
      else if (pad->dualsense && id == 0x31 && got >= 12) decode_sony(buf.data() + 2, true, s);                 // DS5 BT extended
      else if (pad->dualsense && pad->is_bt && id == 0x01 && got >= 10) decode_sony(buf.data() + 1, false, s);  // DS5 BT simple
      else if (!pad->dualsense && id == 0x11 && got >= 12) decode_sony(buf.data() + 3, false, s);               // DS4 BT
      else if (!pad->dualsense && id == 0x01 && got >= 10) decode_sony(buf.data() + 1, false, s);               // DS4 USB
      else continue;
    } else {
      if (!decode_generic(*pad, buf.data(), got, s)) continue;
    }
    std::lock_guard<std::mutex> lk(pad->m);
    pad->state = s;
  }
  CloseHandle(ov.hEvent);
  pad->alive = false;
  std::lock_guard<std::mutex> lk(pad->m);
  pad->state = PadState{};
}

// Finds Sony & generic USB/Bluetooth HID gamepads that are not open yet.
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
    if (path_contains_ig(path)) continue;  // XInput devices are handled natively by xinput_provider
    {
      std::lock_guard<std::mutex> lk(g_m);
      bool open = false;
      for (auto& p : g_pads) open |= p->alive && p->path == path;
      if (open) continue;
    }

    // Try read+write first (needed for feature reports), fall back to read-only for restrictive BT HID stacks.
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
      h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                      OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    }
    if (h == INVALID_HANDLE_VALUE) continue;

    HIDD_ATTRIBUTES attr{sizeof attr};
    if (!HidD_GetAttributes(h, &attr)) {
      CloseHandle(h);
      continue;
    }

    PHIDP_PREPARSED_DATA ppd = nullptr;
    HIDP_CAPS caps{};
    if (!HidD_GetPreparsedData(h, &ppd) || HidP_GetCaps(ppd, &caps) != HIDP_STATUS_SUCCESS) {
      if (ppd) HidD_FreePreparsedData(ppd);
      CloseHandle(h);
      continue;
    }

    const bool sony = (attr.VendorID == kSony) && (is_dualsense(attr.ProductID) || is_ds4(attr.ProductID));
    const bool generic_pad = (caps.UsagePage == HID_USAGE_PAGE_GENERIC) &&
                             (caps.Usage == HID_USAGE_GENERIC_GAMEPAD || caps.Usage == HID_USAGE_GENERIC_JOYSTICK);
    if (!sony && !generic_pad) {
      HidD_FreePreparsedData(ppd);
      CloseHandle(h);
      continue;
    }

    auto pad = std::make_shared<Pad>();
    pad->path = path;
    pad->h = h;
    pad->vid = attr.VendorID;
    pad->pid = attr.ProductID;
    pad->sony = sony;
    pad->dualsense = sony && is_dualsense(attr.ProductID);
    pad->input_report_len = std::max<USHORT>(64, caps.InputReportByteLength);
    pad->is_bt = (caps.InputReportByteLength >= 78);
    pad->state.connected = true;  // report connected immediately so games detect idle BT pads at boot

    if (sony) {
      // A DualSense on Bluetooth switches to the full report (0x31) once feature report 0x05 is read.
      if (pad->dualsense && pad->is_bt) {
        std::vector<uint8_t> feature(std::max<size_t>(41, caps.FeatureReportByteLength), 0);
        feature[0] = 0x05;
        HidD_GetFeature(h, feature.data(), ULONG(feature.size()));
      }
      HidD_FreePreparsedData(ppd);
    } else {
      pad->ppd = ppd;
      if (caps.NumberInputValueCaps > 0) {
        USHORT vc_len = caps.NumberInputValueCaps;
        std::vector<HIDP_VALUE_CAPS> vcs(vc_len);
        if (HidP_GetValueCaps(HidP_Input, vcs.data(), &vc_len, ppd) == HIDP_STATUS_SUCCESS) {
          auto record_axis = [&](AxisInfo& dst, const HIDP_VALUE_CAPS& vc, USAGE u) {
            if (dst.present) return;
            dst.present = true;
            dst.page = vc.UsagePage;
            dst.usage = u;
            dst.logical_min = vc.LogicalMin;
            dst.logical_max = vc.LogicalMax;
            dst.bit_size = vc.BitSize;
          };
          for (USHORT k = 0; k < vc_len; ++k) {
            const auto& vc = vcs[k];
            const USAGE u_lo = vc.IsRange ? vc.Range.UsageMin : vc.NotRange.Usage;
            const USAGE u_hi = vc.IsRange ? vc.Range.UsageMax : vc.NotRange.Usage;
            for (USAGE u = u_lo; u <= u_hi; ++u) {
              if (vc.UsagePage == HID_USAGE_PAGE_GENERIC) {
                switch (u) {
                  case HID_USAGE_GENERIC_X: record_axis(pad->ax_x, vc, u); break;
                  case HID_USAGE_GENERIC_Y: record_axis(pad->ax_y, vc, u); break;
                  case HID_USAGE_GENERIC_Z: record_axis(pad->ax_z, vc, u); break;
                  case HID_USAGE_GENERIC_RX: record_axis(pad->ax_rx, vc, u); break;
                  case HID_USAGE_GENERIC_RY: record_axis(pad->ax_ry, vc, u); break;
                  case HID_USAGE_GENERIC_RZ: record_axis(pad->ax_rz, vc, u); break;
                  case HID_USAGE_GENERIC_HATSWITCH: record_axis(pad->ax_hat, vc, u); break;
                  default: break;
                }
              } else if (vc.UsagePage == HID_USAGE_PAGE_SIMULATION) {
                if (u == 0xC4) record_axis(pad->ax_accel, vc, u);
                else if (u == 0xC5) record_axis(pad->ax_brake, vc, u);
              }
            }
          }
        }
      }
    }

    {
      std::lock_guard<std::mutex> lk(g_m);
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
    scan();  // initial synchronous scan so controllers connected at launch work immediately
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        scan();
      }
    }).detach();
  });
}

}  // namespace

int sony_pad_count() {
  start_scanner();
  std::lock_guard<std::mutex> lk(g_m);
  int n = 0;
  for (auto& p : g_pads) {
    if (!p->alive) continue;
    std::lock_guard<std::mutex> plk(p->m);
    if (p->state.connected) ++n;
  }
  return n;
}

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
int sony_pad_count() { return 0; }
bool sony_pad_read(int, PadState&) { return false; }
}  // namespace input
#endif
