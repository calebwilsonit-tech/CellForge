// cellGcmSys and cellVideoOut HLE on top of rsx::Rsx.
//
// A game builds RSX commands in a command buffer described by a context
// { u32 begin, u32 end, u32 current, u32 callback_opd }. When `current` reaches
// `end`, the game calls `callback`, which hands the filled segment to the RSX and
// moves the context to the next segment. Struct layouts follow the public libgcm
// documentation.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

#include "hle.h"
#include "lv2.h"
#include "rsx.h"

namespace hle {
namespace {

using lv2::Kernel;
using rsx::Rsx;

constexpr uint32_t CELL_OK = 0;
constexpr uint32_t CELL_GCM_ERROR_FAILURE = 0x802100FF;
constexpr uint32_t CELL_GCM_ERROR_INVALID_VALUE = 0x80210002;
constexpr uint32_t CELL_GCM_ERROR_ADDRESS_OVERWRAP = 0x80210005;

constexpr uint32_t kSegmentSize = 0x8000;  // 32 KB command segments
constexpr uint32_t kFirstSegment = 0x1000; // command buffer starts 4 KB into IO memory

Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }

struct State {
  std::mutex m;
  uint32_t context = 0;       // guest CellGcmContextData
  uint32_t callback_opd = 0;
  uint32_t cmd_begin = 0, cmd_end = 0;  // whole command area (guest addresses)
  uint32_t segment = kSegmentSize;
  uint32_t tile_info = 0, zcull_info = 0;
  uint32_t video_resolution = 2;  // 720p
} g;

uint32_t io_of(uint32_t ea) {
  uint32_t io = 0;
  Rsx::get().ea_to_io(ea, &io);
  return io;
}

// Hands everything written so far to the RSX and moves the context to the next
// segment (wrapping around), leaving a jump behind so the RSX follows.
void advance_segment(uint32_t ctx) {
  Kernel& k = K();
  Rsx& r = Rsx::get();
  const uint32_t current = k.r32(ctx + 8);
  const uint32_t nseg = std::max(1u, (g.cmd_end - g.cmd_begin) / g.segment);
  const uint32_t index = std::min((current - g.cmd_begin) / g.segment, nseg - 1);
  const uint32_t next_begin = g.cmd_begin + ((index + 1) % nseg) * g.segment;
  const uint32_t next_io = io_of(next_begin);

  // The RSX may run up to `current`; the jump placed there takes it to the next segment.
  ppu::st32(k.base(), current, 0x20000000u | next_io);
  k.w32(r.control_addr() + 0, io_of(current));

  // Don't reuse the next segment until the RSX has left it.
  for (;;) {
    const uint32_t get = k.r32(r.control_addr() + 4);
    if (get < next_io || get >= next_io + g.segment || get == io_of(current)) break;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  k.w32(ctx + 0, next_begin);
  k.w32(ctx + 4, next_begin + g.segment - 4);
  k.w32(ctx + 8, next_begin);
}

void gcm_callback(ppu::Context& c) {
  std::lock_guard<std::mutex> lk(g.m);
  advance_segment(a32(c, 0));
  ret(c, CELL_OK);
}

// Appends words to the game's command buffer through its context.
void push_commands(uint32_t ctx, std::initializer_list<uint32_t> words) {
  Kernel& k = K();
  std::lock_guard<std::mutex> lk(g.m);
  if (k.r32(ctx + 8) + 4 * uint32_t(words.size()) > k.r32(ctx + 4)) advance_segment(ctx);
  uint32_t cur = k.r32(ctx + 8);
  for (uint32_t w : words) {
    ppu::st32(k.base(), cur, w);
    cur += 4;
  }
  k.w32(ctx + 8, cur);
  // Like libgcm, a flip also flushes: hand everything written so far to the RSX.
  k.w32(Rsx::get().control_addr() + 0, io_of(cur));
}
constexpr uint32_t header(uint32_t method, uint32_t count) { return (count << 18) | method; }

// ---- init / configuration -----------------------------------------------------
void _cellGcmInitBody(ppu::Context& c) {
  // (CellGcmContextData** context, u32 cmdSize, u32 ioSize, u32 ioAddress)
  Kernel& k = K();
  const uint32_t ctx_pp = a32(c, 0), cmd_size = a32(c, 1), io_size = a32(c, 2), io_addr = a32(c, 3);
  if ((io_addr & 0xFFFFF) || (io_size & 0xFFFFF) || !io_size) return ret(c, CELL_GCM_ERROR_FAILURE);
  if (cmd_size > io_size || cmd_size < kFirstSegment + 0x400) return ret(c, CELL_GCM_ERROR_FAILURE);

  Rsx::get().init(io_addr, io_size);
  std::lock_guard<std::mutex> lk(g.m);
  if (!g.context) {
    g.context = k.hle.alloc(16, 16);
    g.callback_opd = make_opd(gcm_callback);
    g.tile_info = k.hle.alloc(15 * 16, 16);
    g.zcull_info = k.hle.alloc(8 * 32, 16);
  }
  g.cmd_begin = io_addr + kFirstSegment;
  g.cmd_end = io_addr + cmd_size;
  g.segment = std::min(kSegmentSize, g.cmd_end - g.cmd_begin);
  k.w32(g.context + 0, g.cmd_begin);
  k.w32(g.context + 4, g.cmd_begin + g.segment - 4);
  k.w32(g.context + 8, g.cmd_begin);
  k.w32(g.context + 12, g.callback_opd);
  k.w32(ctx_pp, g.context);
  k.logf("cellGcm: initialised (io 0x%08X + 0x%X, command buffer 0x%X)", io_addr, io_size, cmd_size);
  ret(c, CELL_OK);
}

void cellGcmGetConfiguration(ppu::Context& c) {
  // CellGcmConfig: localAddress, ioAddress, localSize, ioSize, memoryFrequency, coreFrequency
  const uint32_t a = a32(c, 0);
  Rsx& r = Rsx::get();
  K().w32(a + 0, rsx::kLocalBase);
  K().w32(a + 4, r.io_address());
  K().w32(a + 8, rsx::kLocalUsable);
  K().w32(a + 12, r.io_size());
  K().w32(a + 16, 650000000);
  K().w32(a + 20, 500000000);
  ret(c, CELL_OK);
}

void cellGcmAddressToOffset(ppu::Context& c) {
  const uint32_t addr = a32(c, 0);
  uint32_t off = 0;
  if (addr >= rsx::kLocalBase && addr < rsx::kLocalBase + rsx::kLocalSize) off = addr - rsx::kLocalBase;
  else if (!Rsx::get().ea_to_io(addr, &off)) return ret(c, CELL_GCM_ERROR_FAILURE);
  K().w32(a32(c, 1), off);
  ret(c, CELL_OK);
}
void cellGcmIoOffsetToAddress(ppu::Context& c) {
  uint32_t ea = 0;
  if (!Rsx::get().io_to_ea(a32(c, 0), &ea)) return ret(c, CELL_GCM_ERROR_FAILURE);
  K().w32(a32(c, 1), ea);
  ret(c, CELL_OK);
}
void cellGcmMapMainMemory(ppu::Context& c) {
  uint32_t io = 0;
  if (!Rsx::get().map_main(a32(c, 0), a32(c, 1), &io)) return ret(c, CELL_GCM_ERROR_FAILURE);
  K().w32(a32(c, 2), io);
  ret(c, CELL_OK);
}
void cellGcmMapEaIoAddress(ppu::Context& c) {
  ret(c, Rsx::get().map_io(a32(c, 0), a32(c, 1), a32(c, 2)) ? CELL_OK : CELL_GCM_ERROR_FAILURE);
}
void cellGcmUnmapEaIoAddress(ppu::Context& c) { ret(c, Rsx::get().unmap_ea(a32(c, 0)) ? CELL_OK : CELL_GCM_ERROR_FAILURE); }
void cellGcmUnmapIoAddress(ppu::Context& c) { ret(c, Rsx::get().unmap_io(a32(c, 0)) ? CELL_OK : CELL_GCM_ERROR_FAILURE); }
void cellGcmGetOffsetTable(ppu::Context& c) {
  K().w32(a32(c, 0) + 0, Rsx::get().io_table_addr());
  K().w32(a32(c, 0) + 4, Rsx::get().ea_table_addr());
  ret(c, CELL_OK);
}
void cellGcmGetMaxIoMapSize(ppu::Context& c) { ret(c, rsx::kMaxIoPages * rsx::kIoPage - Rsx::get().io_size()); }

void cellGcmGetControlRegister(ppu::Context& c) { ret(c, Rsx::get().control_addr()); }
void cellGcmGetLabelAddress(ppu::Context& c) { ret(c, Rsx::get().label_addr(a32(c, 0) & 0xFF)); }
void cellGcmGetReportDataAddress(ppu::Context& c) { ret(c, Rsx::get().report_addr(a32(c, 0))); }
void cellGcmGetReportDataAddressLocation(ppu::Context& c) { ret(c, Rsx::get().report_addr(a32(c, 0))); }
void cellGcmGetReportDataLocation(ppu::Context& c) { ret(c, K().r32(Rsx::get().report_addr(a32(c, 0)) + 8)); }
void cellGcmGetTimeStamp(ppu::Context& c) { ret(c, K().r64(Rsx::get().report_addr(a32(c, 0)))); }
void cellGcmGetTimeStampLocation(ppu::Context& c) { ret(c, K().r64(Rsx::get().report_addr(a32(c, 0)))); }

void cellGcmGetTiledPitchSize(ppu::Context& c) {
  static const uint32_t kPitches[] = {0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800, 0xA00, 0xC00,
                                      0xD00, 0xE00, 0x1000, 0x1400, 0x1800, 0x1A00, 0x1C00, 0x2000,
                                      0x2800, 0x3000, 0x3400, 0x3800, 0x4000, 0x5000, 0x6000, 0x6800,
                                      0x7000, 0x8000, 0xA000, 0xC000, 0xD000, 0xE000, 0x10000};
  const uint32_t size = a32(c, 0);
  for (uint32_t p : kPitches)
    if (p >= size) return ret(c, p);
  ret(c, 0);
}
void cellGcmGetDefaultCommandWordSize(ppu::Context& c) { ret(c, 0x400); }
void cellGcmGetDefaultSegmentWordSize(ppu::Context& c) { ret(c, 0x100); }
void cellGcmGetTileInfo(ppu::Context& c) { ret(c, g.tile_info); }
void cellGcmGetZcullInfo(ppu::Context& c) { ret(c, g.zcull_info); }

// ---- display & flips -------------------------------------------------------------
void cellGcmSetDisplayBuffer(ppu::Context& c) {
  // (u8 id, u32 offset, u32 pitch, u32 width, u32 height)
  if (a32(c, 0) >= 8) return ret(c, CELL_GCM_ERROR_INVALID_VALUE);
  Rsx::get().set_display_buffer(a32(c, 0), a32(c, 1), a32(c, 2), a32(c, 3), a32(c, 4));
  ret(c, CELL_OK);
}
void cellGcmGetDisplayInfo(ppu::Context& c) { ret(c, Rsx::get().display_info_addr()); }
void cellGcmGetFlipStatus(ppu::Context& c) { ret(c, Rsx::get().flip_status.load()); }
void cellGcmResetFlipStatus(ppu::Context& c) { Rsx::get().flip_status = 1; ret(c, CELL_OK); }
void cellGcmSetFlipHandler(ppu::Context& c) { Rsx::get().flip_handler = a32(c, 0); ret(c, CELL_OK); }
void cellGcmSetVBlankHandler(ppu::Context& c) { Rsx::get().vblank_handler = a32(c, 0); ret(c, CELL_OK); }
void cellGcmGetLastFlipTime(ppu::Context& c) { ret(c, Rsx::get().last_flip_time.load()); }
void cellGcmGetVBlankCount(ppu::Context& c) { ret(c, Rsx::get().stats.vblanks.load()); }
void cellGcmGetCurrentField(ppu::Context& c) { ret(c, 0); }

void cellGcmSetFlipCommand(ppu::Context& c) {
  // (CellGcmContextData* ctx, u32 id)
  const uint32_t ctx = a32(c, 0), id = a32(c, 1);
  if (id >= 8) return ret(c, CELL_GCM_ERROR_INVALID_VALUE);
  push_commands(ctx, {header(rsx::kMethodFlip, 1), id});
  ret(c, CELL_OK);
}
void cellGcmSetFlipCommandWithWaitLabel(ppu::Context& c) {
  // (ctx, u32 id, u8 label_index, u32 label_value)
  const uint32_t ctx = a32(c, 0), id = a32(c, 1);
  if (id >= 8) return ret(c, CELL_GCM_ERROR_INVALID_VALUE);
  push_commands(ctx, {header(rsx::kMethodWaitLabel, 2), a32(c, 2) & 0xFF, a32(c, 3), header(rsx::kMethodFlip, 1), id});
  ret(c, CELL_OK);
}
void cellGcmSetPrepareFlip(ppu::Context& c) {
  const uint32_t ctx = a32(c, 0), id = a32(c, 1);
  if (id >= 8) return ret(c, CELL_GCM_ERROR_INVALID_VALUE);
  push_commands(ctx, {header(rsx::kMethodFlip, 1), id});
  ret(c, id);
}
void cellGcmSetFlipImmediate(ppu::Context& c) {
  if (a32(c, 0) >= 8) return ret(c, CELL_GCM_ERROR_INVALID_VALUE);
  Rsx::get().flip(a32(c, 0), true);
  ret(c, CELL_OK);
}
void cellGcmSetFlipMode(ppu::Context& c) {
  const uint32_t mode = a32(c, 0);
  if (mode == 1 || mode == 2) Rsx::get().flip_mode = mode;
  K().logf("cellGcmSetFlipMode(%u)", mode);
  ret(c, CELL_OK);
}
void cellGcmSetWaitFlip(ppu::Context& c) { ret(c, CELL_OK); }  // our flips complete in order

void gcm_ok(ppu::Context& c) { ret(c, CELL_OK); }

// ============================================================================
// cellVideoOut (in cellSysutil)
// ============================================================================
bool resolution_size(uint32_t id, uint16_t& w, uint16_t& h) {
  switch (id) {
    case 1: w = 1920; h = 1080; return true;
    case 2: w = 1280; h = 720; return true;
    case 4: w = 720; h = 480; return true;
    case 5: w = 720; h = 576; return true;
    default: return false;
  }
}

void cellVideoOutGetState(ppu::Context& c) {
  // (u32 videoOut, u32 deviceIndex, CellVideoOutState* state)
  // state: u8 state, u8 colorSpace, u8 reserved[6],
  //        displayMode { u8 resolutionId, u8 scanMode, u8 conversion, u8 aspect, u8 reserved[2], u16 refreshRates }
  const uint32_t a = a32(c, 2);
  uint8_t* p = K().base() + a;
  std::memset(p, 0, 16);
  p[0] = 0;                        // enabled
  p[1] = 1;                        // RGB
  p[8] = uint8_t(g.video_resolution);
  p[9] = 1;                        // progressive
  p[11] = 2;                       // 16:9
  ppu::st16(K().base(), a + 14, 1);  // 59.94 Hz
  ret(c, CELL_OK);
}
void cellVideoOutGetResolution(ppu::Context& c) {
  uint16_t w = 0, h = 0;
  if (!resolution_size(a32(c, 0), w, h)) return ret(c, CELL_GCM_ERROR_INVALID_VALUE);
  ppu::st16(K().base(), a32(c, 1), w);
  ppu::st16(K().base(), a32(c, 1) + 2, h);
  ret(c, CELL_OK);
}
void cellVideoOutGetResolutionAvailability(ppu::Context& c) {
  // (videoOut, resolutionId, aspect, option) -> 1 if available
  const uint32_t id = a32(c, 1);
  ret(c, id == 1 || id == 2 ? 1 : 0);
}
void cellVideoOutConfigure(ppu::Context& c) {
  // (videoOut, CellVideoOutConfiguration* config, option, waitForEvent)
  const uint32_t id = K().base()[a32(c, 1)];
  uint16_t w, h;
  if (resolution_size(id, w, h)) g.video_resolution = id;
  ret(c, CELL_OK);
}
void cellVideoOutGetConfiguration(ppu::Context& c) {
  // config: u8 resolutionId, u8 format, u8 aspect, u8 reserved[9], u32 pitch
  const uint32_t a = a32(c, 1);
  uint16_t w = 1280, h = 720;
  resolution_size(g.video_resolution, w, h);
  std::memset(K().base() + a, 0, 16);
  K().base()[a + 0] = uint8_t(g.video_resolution);
  K().base()[a + 1] = 0;  // X8R8G8B8
  K().base()[a + 2] = 2;  // 16:9
  K().w32(a + 12, uint32_t(w) * 4);
  ret(c, CELL_OK);
}
void cellVideoOutGetNumberOfDevice(ppu::Context& c) { ret(c, 1); }

}  // namespace

void register_gcm_modules() {
#define R(fn) register_function("cellGcmSys", #fn, fn)
  R(_cellGcmInitBody); R(cellGcmGetConfiguration); R(cellGcmAddressToOffset); R(cellGcmIoOffsetToAddress);
  R(cellGcmMapMainMemory); R(cellGcmMapEaIoAddress); R(cellGcmUnmapEaIoAddress); R(cellGcmUnmapIoAddress);
  R(cellGcmGetOffsetTable); R(cellGcmGetMaxIoMapSize); R(cellGcmGetControlRegister); R(cellGcmGetLabelAddress);
  R(cellGcmGetReportDataAddress); R(cellGcmGetReportDataAddressLocation); R(cellGcmGetReportDataLocation);
  R(cellGcmGetTimeStamp); R(cellGcmGetTimeStampLocation); R(cellGcmGetTiledPitchSize);
  R(cellGcmGetDefaultCommandWordSize); R(cellGcmGetDefaultSegmentWordSize); R(cellGcmGetTileInfo);
  R(cellGcmGetZcullInfo); R(cellGcmSetDisplayBuffer); R(cellGcmGetDisplayInfo); R(cellGcmGetFlipStatus);
  R(cellGcmResetFlipStatus); R(cellGcmSetFlipHandler); R(cellGcmSetVBlankHandler); R(cellGcmGetLastFlipTime);
  R(cellGcmGetVBlankCount); R(cellGcmGetCurrentField); R(cellGcmSetFlipCommand);
  R(cellGcmSetFlipCommandWithWaitLabel); R(cellGcmSetPrepareFlip); R(cellGcmSetFlipImmediate); R(cellGcmSetFlipMode); R(cellGcmSetWaitFlip);
#undef R
  register_function("cellGcmSys", "_cellGcmSetFlipCommand", cellGcmSetFlipCommand);
  register_function("cellGcmSys", "_cellGcmSetFlipCommandWithWaitLabel", cellGcmSetFlipCommandWithWaitLabel);
  register_function("cellGcmSys", "cellGcmSetFlip", cellGcmSetFlipCommand);
  register_function("cellGcmSys", "cellGcmMapEaIoAddressWithFlags", cellGcmMapEaIoAddress);
  register_function("cellGcmSys", "cellGcmCallback", gcm_callback);
  for (const char* n : {"cellGcmInitDefaultFifoMode", "cellGcmSetDefaultFifoSize",
                        "cellGcmSetVBlankFrequency", "cellGcmSetSecondVFrequency", "cellGcmSetDebugOutputLevel",
                        "cellGcmInitSystemMode", "cellGcmSetGraphicsHandler", "cellGcmSetQueueHandler",
                        "cellGcmSetUserHandler", "cellGcmSetSecondVHandler", "cellGcmSetZcull", "cellGcmBindZcull",
                        "cellGcmUnbindZcull", "cellGcmSetInvalidateTile", "cellGcmBindTile", "cellGcmUnbindTile",
                        "cellGcmSetTile", "cellGcmSetTileInfo", "cellGcmReserveIoMapSize",
                        "cellGcmUnreserveIoMapSize", "cellGcmSortRemapEaIoAddress", "cellGcmDumpGraphicsError"})
    register_function("cellGcmSys", n, gcm_ok);

#define V(fn) register_function("cellSysutil", #fn, fn)
  V(cellVideoOutGetState); V(cellVideoOutGetResolution); V(cellVideoOutGetResolutionAvailability);
  V(cellVideoOutConfigure); V(cellVideoOutGetConfiguration); V(cellVideoOutGetNumberOfDevice);
#undef V
}

}  // namespace hle
