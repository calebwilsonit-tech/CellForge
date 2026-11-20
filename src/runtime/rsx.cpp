#define _CRT_SECURE_NO_WARNINGS  // getenv/fopen for the debug trace and frame-dump options
#include "rsx.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "lv2.h"
#include "options.h"
#include "profiler.h"
#include "vk_renderer.h"

namespace rsx {
namespace {

using Clock = std::chrono::steady_clock;

lv2::Kernel& K() { return lv2::Kernel::get(); }

// FIFO command word encodings.
constexpr uint32_t kOldJumpMask = 0xE0000003, kOldJump = 0x20000000;
constexpr uint32_t kReturn = 0x00020000;
constexpr uint32_t kNonIncrement = 0x40000000;

// Methods handled in phase 1 (NV406E = FIFO channel, NV4097 = 3D engine).
constexpr uint32_t M_SET_REFERENCE = 0x0050;
constexpr uint32_t M_SEMAPHORE_OFFSET = 0x0064;
constexpr uint32_t M_SEMAPHORE_ACQUIRE = 0x0068;
constexpr uint32_t M_SEMAPHORE_RELEASE = 0x006C;
constexpr uint32_t M_CONTEXT_DMA_COLOR_A = 0x0194;
constexpr uint32_t M_SURFACE_CLIP_H = 0x0200;
constexpr uint32_t M_SURFACE_CLIP_V = 0x0204;
constexpr uint32_t M_SURFACE_PITCH_A = 0x020C;
constexpr uint32_t M_SURFACE_COLOR_AOFFSET = 0x0210;
constexpr uint32_t M_BEGIN_END = 0x1808;
constexpr uint32_t M_SEMAPHORE_OFFSET_3D = 0x1D6C;
constexpr uint32_t M_BACK_END_SEMAPHORE_RELEASE = 0x1D70;
constexpr uint32_t M_TEXTURE_SEMAPHORE_RELEASE = 0x1D74;
constexpr uint32_t M_COLOR_CLEAR_VALUE = 0x1D90;
constexpr uint32_t M_CLEAR_SURFACE = 0x1D94;
constexpr uint32_t M_ZPASS_PIXEL_COUNT_ENABLE = 0x1DEC;
constexpr uint32_t M_CLEAR_REPORT_VALUE = 0x1DFC;
constexpr uint32_t M_GET_REPORT = 0x1E00;

constexpr uint32_t kDmaMain = 0xFEED0001;

// PS3RECOMP_TRACE_FLIP=<spec> accepts comma-separated entries (e.g. "8900+,10600+,11500+").
// Each N traces flip N; N+ latches the first flip >= N with a depth-tested 3D draw.
// A single N+ traces l and l+1; multiple comma entries trace 1 flip per entry.
bool trace_flip_active(uint64_t flip, bool is_3d_draw) {
  struct Spec {
    long long target = -1;
    bool plus = false;
  };
  struct List {
    Spec items[8]{};
    int count = 0;
  };
  static const List list = [] {
    List l;
    if (const char* e = std::getenv("PS3RECOMP_TRACE_FLIP")) {
      const char* p = e;
      while (*p && l.count < 8) {
        char* end = nullptr;
        long long t = std::strtoll(p, &end, 10);
        if (end == p) break;
        l.items[l.count].target = t;
        l.items[l.count].plus = end && *end == '+';
        if (end && *end == '+') ++end;
        ++l.count;
        p = (end && *end == ',') ? end + 1 : "";
      }
    }
    return l;
  }();
  if (list.count == 0) return false;
  static std::atomic<long long> latched[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  for (int i = 0; i < list.count; ++i) {
    const Spec& s = list.items[i];
    if (!s.plus) {
      if (static_cast<long long>(flip) == s.target) return true;
      continue;
    }
    long long l = latched[i].load(std::memory_order_relaxed);
    if (l < 0 && static_cast<long long>(flip) >= s.target && is_3d_draw) {
      l = static_cast<long long>(flip);
      latched[i].store(l, std::memory_order_relaxed);
      K().logf("rsx: trace_flip latched[%d] on flip %llu", i, (unsigned long long)flip);
    }
    if (l >= 0 && (static_cast<long long>(flip) == l || (list.count == 1 && static_cast<long long>(flip) == l + 1)))
      return true;
  }
  return false;
}

}  // namespace

Rsx& Rsx::get() {
  static Rsx r;
  return r;
}

Rsx::Rsx() = default;
Rsx::~Rsx() {
  shutdown();
  gpu_.reset();
}

GpuRenderer* Rsx::gpu() {
  std::lock_guard<std::mutex> lk(gpu_m_);
  if (!use_gpu) return nullptr;
  if (!gpu_tried_) {
    gpu_tried_ = true;
    gpu_ = GpuRenderer::create(K().log);
  }
  return gpu_.get();
}

void Rsx::poll_gpu() {
  GpuRenderer* g = nullptr;
  {
    std::lock_guard<std::mutex> lk(gpu_m_);
    g = gpu_.get();
  }
  if (g) g->poll();
}

void Rsx::sync(bool include_depth) {
  GpuRenderer* g = nullptr;
  {
    std::lock_guard<std::mutex> lk(gpu_m_);
    g = gpu_.get();
  }
  if (g) g->flush(K().base(), include_depth);
}

std::string Rsx::gpu_name() {
  GpuRenderer* g = gpu();
  return g ? g->device_name() : std::string();
}

void Rsx::init(uint32_t io_address, uint32_t io_size) {
  lv2::Kernel& k = K();
  if (!initialized_) {
    k.memory().map(kLocalBase, kLocalSize);
    control_ = k.hle.alloc(16, 16);
    labels_ = k.hle.alloc(256 * 16, 16);
    if (std::getenv("PS3RECOMP_RSX_TRACE")) k.logf("rsx trace: labels at 0x%08X", labels_);
    io_table_ = k.hle.alloc(kEaPages * 2, 16);
    ea_table_ = k.hle.alloc(kMaxIoPages * 2, 16);
    display_info_ = k.hle.alloc(8 * 16, 16);
  }
  {
    std::lock_guard<std::mutex> lk(map_m_);
    for (uint32_t i = 0; i < kEaPages; ++i) {
      io_of_ea_[i] = kIoUnmapped;
      ppu::st16(k.base(), io_table_ + 2 * i, kIoUnmapped);
    }
    for (uint32_t i = 0; i < kMaxIoPages; ++i) {
      ea_of_io_[i] = kIoUnmapped;
      ppu::st16(k.base(), ea_table_ + 2 * i, kIoUnmapped);
    }
  }
  io_address_ = io_address;
  io_size_ = io_size;
  map_io(io_address, 0, io_size);
  // libgcm places the first command segment 4 KB into the IO area.
  k.w32(control_ + 0, 0x1000);
  k.w32(control_ + 4, 0x1000);
  k.w32(control_ + 8, 0);
  flip_status = 0;
  std::fill(regs_.begin(), regs_.end(), 0);
  reset_defaults(regs_.data());

  if (!presenter_ && presenter_factory) set_presenter(presenter_factory());
  if (!initialized_) {
    initialized_ = true;
    stop_ = false;
    puller_ = std::thread([this] { prof::register_thread("rsx:puller"); run(); });
    vblank_ = std::thread([this] { vblank_loop(); });
  }
}

void Rsx::shutdown() {
  stop_ = true;
  vblank_cv_.notify_all();
  if (puller_.joinable()) puller_.join();
  if (vblank_.joinable()) vblank_.join();
  initialized_ = false;
}

// ---- IO mapping ------------------------------------------------------------------
bool Rsx::map_io(uint32_t ea, uint32_t io, uint32_t size) {
  if ((ea | io | size) & (kIoPage - 1) || !size) return false;
  const uint32_t ea_page = ea / kIoPage, io_page = io / kIoPage, pages = size / kIoPage;
  if (ea_page + pages > kEaPages || io_page + pages > kMaxIoPages) return false;
  std::lock_guard<std::mutex> lk(map_m_);
  for (uint32_t i = 0; i < pages; ++i)
    if (ea_of_io_[io_page + i] != kIoUnmapped || io_of_ea_[ea_page + i] != kIoUnmapped) return false;
  for (uint32_t i = 0; i < pages; ++i) {
    io_of_ea_[ea_page + i] = uint16_t(io_page + i);
    ea_of_io_[io_page + i] = uint16_t(ea_page + i);
    ppu::st16(K().base(), io_table_ + 2 * (ea_page + i), uint16_t(io_page + i));
    ppu::st16(K().base(), ea_table_ + 2 * (io_page + i), uint16_t(ea_page + i));
  }
  return true;
}

bool Rsx::map_main(uint32_t ea, uint32_t size, uint32_t* io_out) {
  if ((ea | size) & (kIoPage - 1) || !size) return false;
  const uint32_t pages = size / kIoPage;
  uint32_t start = 0;
  {
    std::lock_guard<std::mutex> lk(map_m_);
    uint32_t run = 0;
    bool found = false;
    for (uint32_t p = 0; p < kMaxIoPages && !found; ++p) {
      run = ea_of_io_[p] == kIoUnmapped ? run + 1 : 0;
      if (run == pages) { start = p + 1 - pages; found = true; }
    }
    if (!found) return false;
  }
  if (!map_io(ea, start * kIoPage, size)) return false;
  *io_out = start * kIoPage;
  return true;
}

bool Rsx::unmap_ea(uint32_t ea) {
  std::lock_guard<std::mutex> lk(map_m_);
  uint32_t p = ea / kIoPage;
  if (p >= kEaPages || io_of_ea_[p] == kIoUnmapped) return false;
  // Unmap the contiguous run that starts here.
  while (p < kEaPages && io_of_ea_[p] != kIoUnmapped) {
    const uint16_t io = io_of_ea_[p];
    ea_of_io_[io] = kIoUnmapped;
    io_of_ea_[p] = kIoUnmapped;
    ppu::st16(K().base(), io_table_ + 2 * p, kIoUnmapped);
    ppu::st16(K().base(), ea_table_ + 2 * io, kIoUnmapped);
    ++p;
    if (p < kEaPages && io_of_ea_[p] != uint16_t(io + 1)) break;
  }
  return true;
}

bool Rsx::unmap_io(uint32_t io) {
  uint32_t ea = 0;
  return io_to_ea(io, &ea) && unmap_ea(ea & ~(kIoPage - 1));
}

bool Rsx::ea_to_io(uint32_t ea, uint32_t* io) const {
  std::lock_guard<std::mutex> lk(map_m_);
  const uint32_t p = ea / kIoPage;
  if (p >= kEaPages || io_of_ea_[p] == kIoUnmapped) return false;
  *io = uint32_t(io_of_ea_[p]) * kIoPage + (ea & (kIoPage - 1));
  return true;
}

bool Rsx::io_to_ea(uint32_t io, uint32_t* ea) const {
  std::lock_guard<std::mutex> lk(map_m_);
  const uint32_t p = io / kIoPage;
  if (p >= kMaxIoPages || ea_of_io_[p] == kIoUnmapped) return false;
  *ea = uint32_t(ea_of_io_[p]) * kIoPage + (io & (kIoPage - 1));
  return true;
}

uint32_t Rsx::offset_to_address(uint32_t offset, uint32_t location) const {
  if (location == 0) return kLocalBase + (offset & (kLocalSize - 1));
  uint32_t ea = 0;
  return io_to_ea(offset, &ea) ? ea : 0;
}

uint32_t Rsx::read_io32(uint32_t io) const {
  uint32_t ea = 0;
  if (!io_to_ea(io, &ea)) return 0;
  return ppu::ld32(K().base(), ea);
}

// ---- display ---------------------------------------------------------------------
void Rsx::set_display_buffer(uint32_t id, uint32_t offset, uint32_t pitch, uint32_t width, uint32_t height) {
  if (id >= 8) return;
  displays_[id] = DisplayBuffer{offset, pitch, width, height, true};
  K().w32(display_info_ + 16 * id + 0, offset);
  K().w32(display_info_ + 16 * id + 4, pitch);
  K().w32(display_info_ + 16 * id + 8, width);
  K().w32(display_info_ + 16 * id + 12, height);
}

void Rsx::set_presenter(std::shared_ptr<gfx::Presenter> p) {
  std::lock_guard<std::mutex> lk(present_m_);
  presenter_ = std::move(p);
}

void Rsx::call_handler(uint32_t opd, uint64_t arg) {
  if (!opd) return;
  try {
    K().call_guest(opd, {arg});
  } catch (const std::exception& e) {
    K().logf("rsx: guest handler 0x%08X failed: %s", opd, e.what());
  }
}

void Rsx::flip(uint32_t id, bool immediate) {
  // Showing the frame (reading the display buffer from guest memory) happens once
  // the GPU has finished it; the flip itself completes at VBlank (in VSYNC mode),
  // pacing lightweight video/menu frames smoothly to 59.94 Hz without stalling
  // frames that already took longer than one VBlank interval.
  const uint64_t flip_no = stats.flips.load() + 1;
  if (id < 8 && displays_[id].valid) {
    const DisplayBuffer d = displays_[id];
    const uint32_t base = kLocalBase + d.offset;
    const uint32_t pitch_px = d.pitch / 4;
    if (d.width && d.height && pitch_px >= d.width && uint64_t(d.offset) + uint64_t(d.pitch) * d.height <= kLocalSize) {
      auto show = [this, d, base, flip_no] { present_display(d, base, flip_no); };
      if (GpuRenderer* g = gpu()) g->flip_async(K().base(), base, d.pitch * d.height, show);
      else show();
    } else if (GpuRenderer* g = gpu()) {
      g->flip_async(K().base(), 0, 0, nullptr);
    }
  } else if (GpuRenderer* g = gpu()) {
    g->flip_async(K().base(), 0, 0, nullptr);
  }
  const int vsync_opt = cfg::g_vsync_mode.load(std::memory_order_relaxed);
  const int cap = cfg::g_fps_cap.load(std::memory_order_relaxed);
  // Respect user V-Sync Off / >60 FPS / Uncapped settings instead of forcing 60 Hz VBlank stalls.
  const bool vblank_paced = !immediate && flip_mode.load() == 2 && !stop_ && vsync_opt != 0 && (cap == 60);
  if (vblank_paced) {
    uint64_t vb = stats.vblanks.load();
    if (vb == last_flip_vblank_) {
      poll_gpu();
      std::unique_lock<std::mutex> lk(vblank_m_);
      vblank_cv_.wait(lk, [&] { return stop_.load() || stats.vblanks.load() != last_flip_vblank_; });
      vb = stats.vblanks.load();
    }
    poll_gpu();
    last_flip_vblank_ = vb;
  }
  // Only apply the software FPS cap when VBlank isn't already pacing at 60 Hz;
  // use a high-resolution timer on Windows so sleeps don't quantize to 15.6 ms ticks.
  if (cap > 0 && !(vblank_paced && cap == 60) && !stop_) {
    const uint64_t target_us = 1000000ull / uint64_t(cap);
    const uint64_t now_us = lv2::system_time_us();
    if (last_flip_time > 0 && now_us < last_flip_time + target_us) {
      const uint64_t wait_us = (last_flip_time + target_us) - now_us;
#if defined(_WIN32)
      static thread_local HANDLE cap_timer = [] {
        HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /*HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
        return h ? h : CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
      }();
      LARGE_INTEGER due;
      due.QuadPart = -int64_t(std::max<uint64_t>(1, wait_us * 10));
      if (cap_timer && SetWaitableTimer(cap_timer, &due, 0, nullptr, nullptr, FALSE))
        WaitForSingleObject(cap_timer, INFINITE);
      else
        std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
#else
      std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
#endif
    }
    const uint64_t after_us = lv2::system_time_us();
    last_flip_time = (last_flip_time > 0 && after_us < last_flip_time + target_us * 2)
                         ? (last_flip_time + target_us)
                         : after_us;
  } else {
    last_flip_time = lv2::system_time_us();
  }
  ++stats.flips;
  flip_status = 0;
  call_handler(flip_handler, 1);
}

// Copies a finished display buffer out of guest memory to the window (and to a
// BMP when PS3RECOMP_DUMP_FRAME=<file.bmp>[,<flip number>] asks for this flip).
void Rsx::present_display(const DisplayBuffer& d, uint32_t base, uint64_t flip_no) {
  std::shared_ptr<gfx::Presenter> p;
  {
    std::lock_guard<std::mutex> lk(present_m_);
    p = presenter_;
  }
  if (p) {
    static uint32_t applied_gen = 0;
    const uint32_t gen = cfg::g_settings_gen.load(std::memory_order_acquire);
    if (gen != applied_gen) {
      applied_gen = gen;
      p->configure(cfg::g_window_width.load(), cfg::g_window_height.load(),
                   cfg::g_display_mode.load(), cfg::g_vsync_mode.load(),
                   cfg::g_upscale_filter.load(), cfg::g_fps_counter_enabled.load());
    }
  }
  static const char* dump = std::getenv("PS3RECOMP_DUMP_FRAME");
  std::string dump_file;
  uint64_t dump_when = 0;
  if (dump) {
    std::string spec(dump);
    dump_file = spec;
    dump_when = 30;
    if (auto comma = spec.rfind(','); comma != std::string::npos) {
      dump_file = spec.substr(0, comma);
      dump_when = std::strtoull(spec.c_str() + comma + 1, nullptr, 10);
    }
  }
  bool dumping_now = dump && flip_no == dump_when;
  // PS3RECOMP_DUMP_EVERY=<folder>,<N>: save every Nth flip as <folder>\flip_<n>.bmp.
  static const char* every = std::getenv("PS3RECOMP_DUMP_EVERY");
  if (every) {
    std::string spec(every);
    const auto comma = spec.rfind(',');
    const uint64_t n = comma == std::string::npos ? 300 : std::strtoull(spec.c_str() + comma + 1, nullptr, 10);
    if (n && flip_no % n == 0) {
      char name[64];
      std::snprintf(name, sizeof name, "\\flip_%06llu.bmp", (unsigned long long)flip_no);
      dump_file = spec.substr(0, comma) + name;
      dump_when = flip_no;
      dumping_now = true;
    }
  }
  if (!p && !dumping_now) return;
  frame_.resize(size_t(d.width) * d.height);
  // Guest pixels are A8R8G8B8 stored big-endian; as a host u32 B,G,R,A that is
  // simply the loaded value with alpha forced opaque.
  const uint8_t* mem = K().base();
  for (uint32_t y = 0; y < d.height; ++y)
    for (uint32_t x = 0; x < d.width; ++x)
      frame_[size_t(y) * d.width + x] = ppu::ld32(mem, base + y * d.pitch + x * 4) | 0xFF000000u;
  if (dumping_now) {
    if (std::FILE* f = std::fopen(dump_file.c_str(), "wb")) {
      const uint32_t w = d.width, h = d.height, img = w * h * 4;
      auto u16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
      auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
      std::fwrite("BM", 1, 2, f); u32(54 + img); u32(0); u32(54);
      u32(40); u32(w); u32(uint32_t(-int32_t(h))); u16(1); u16(32); u32(0); u32(img); u32(2835); u32(2835); u32(0); u32(0);
      std::fwrite(frame_.data(), 1, img, f);  // host BGRA = BMP's 32-bit layout, top-down
      std::fclose(f);
      K().logf("rsx: saved flip %llu to %s", (unsigned long long)dump_when, dump_file.c_str());
    }
  }
  if (p) {
    p->present(frame_.data(), d.width, d.height, d.width);
    if (!p->alive()) {
      K().logf("window closed, exiting");
      std::fflush(nullptr);
      std::_Exit(0);
    }
  }
}
// ---- surface clears ----------------------------------------------------------------
void Rsx::clear_surface(uint32_t mask) {
  ++stats.clears;
  const uint32_t color_bits = mask & 0xF0;
  if (!color_bits) return;  // depth/stencil only: nothing to show yet

  uint32_t aa_x = 1, aa_y = 1;
  surface_aa_scale(regs_[m::SURFACE_FORMAT], aa_x, aa_y);
  const uint32_t clip_h = regs_[M_SURFACE_CLIP_H / 4], clip_v = regs_[M_SURFACE_CLIP_V / 4];
  const uint32_t x0 = (clip_h & 0xFFFF) * aa_x, w = (clip_h >> 16) * aa_x;
  const uint32_t y0 = (clip_v & 0xFFFF) * aa_y, h = (clip_v >> 16) * aa_y;
  const uint32_t pitch = regs_[M_SURFACE_PITCH_A / 4], offset = regs_[M_SURFACE_COLOR_AOFFSET / 4];
  if (!w || !h || pitch < (x0 + w) * 4) return;  // only 32-bit colour surfaces in phase 1
  const uint32_t location = regs_[M_CONTEXT_DMA_COLOR_A / 4] == kDmaMain ? 1 : 0;
  const uint32_t base = offset_to_address(offset, location);
  if (!base) return;
  if (location == 0 && uint64_t(offset) + uint64_t(pitch) * (y0 + h) > kLocalSize) return;

  // Channel write mask: A = 0x80, R = 0x10, G = 0x20, B = 0x40.
  uint32_t write = 0;
  if (mask & 0x80) write |= 0xFF000000u;
  if (mask & 0x10) write |= 0x00FF0000u;
  if (mask & 0x20) write |= 0x0000FF00u;
  if (mask & 0x40) write |= 0x000000FFu;
  const uint32_t value = regs_[M_COLOR_CLEAR_VALUE / 4];
  uint8_t* mem = K().base();
  for (uint32_t y = y0; y < y0 + h; ++y) {
    for (uint32_t x = x0; x < x0 + w; ++x) {
      const uint32_t a = base + y * pitch + x * 4;
      const uint32_t old = write == 0xFFFFFFFFu ? 0 : ppu::ld32(mem, a);
      ppu::st32(mem, a, (old & ~write) | (value & write));
    }
  }
}

// ---- command execution -------------------------------------------------------------
void Rsx::execute(uint32_t method, uint32_t value) {
  regs_[(method & 0xFFFC) / 4] = value;
  lv2::Kernel& k = K();
  static const bool trace = std::getenv("PS3RECOMP_RSX_TRACE") != nullptr;
  if (trace) {
    static std::vector<bool> seen(0x10000 / 4);
    if (!seen[(method & 0xFFFC) / 4]) {
      seen[(method & 0xFFFC) / 4] = true;
      k.logf("rsx trace: first method 0x%04X = 0x%08X", method, value);
    }
  }
  switch (method) {
    case M_SET_REFERENCE:
      k.w32(control_ + 8, value);
      break;
    case M_SEMAPHORE_OFFSET:
      semaphore_offset_406e_ = value;
      break;
    case M_SEMAPHORE_ACQUIRE: {
      const auto t0 = Clock::now();
      for (uint32_t spin = 0; !stop_ && k.r32(labels_ + semaphore_offset_406e_) != value; ++spin) {
        if ((spin & 63) == 0) poll_gpu();
        spin < 4000 ? std::this_thread::yield() : std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (prof::enabled())
        prof::record_call(10001, semaphore_offset_406e_ / 16,
                          uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()), "rsx:acquire");
      break;
    }
    case M_SEMAPHORE_RELEASE: {
      const auto t0 = Clock::now();
      k.w32(labels_ + semaphore_offset_406e_, value);
      if (prof::enabled())
        prof::record_call(10002, semaphore_offset_406e_ / 16,
                          uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()), "rsx:release");
      break;
    }
    case M_SEMAPHORE_OFFSET_3D:
      semaphore_offset_4097_ = value;
      break;
    case M_BACK_END_SEMAPHORE_RELEASE:
      if (prof::enabled()) prof::record_call(10003, semaphore_offset_4097_ / 16, 0, "rsx:backend-release");
      // The back-end release swaps the red and blue bytes of the value.
      k.w32(labels_ + semaphore_offset_4097_,
            (value & 0xFF00FF00u) | ((value & 0xFFu) << 16) | ((value >> 16) & 0xFFu));
      break;
    case M_TEXTURE_SEMAPHORE_RELEASE:
      k.w32(labels_ + semaphore_offset_4097_, value);
      break;
    case M_ZPASS_PIXEL_COUNT_ENABLE:
      break;
    case M_CLEAR_REPORT_VALUE:
      zpass_pixels_ = 0;
      break;
    case M_GET_REPORT: {
      const uint32_t type = value >> 24;
      const uint32_t byte_off = (value & 0xFFFFFFu) & ~15u;
      const uint64_t ts = std::max<uint64_t>(
          1u, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count()));
      const uint32_t val = (type == 1) ? std::max<uint32_t>(zpass_pixels_, 64u) : 0u;
      auto write_report = [&](uint32_t addr) {
        if (!addr) return;
        k.w64(addr + 0, ts);
        k.w32(addr + 8, val);
        k.w32(addr + 12, 0);
      };
      write_report(report_addr(byte_off / 16));
      break;
    }
    case M_CLEAR_SURFACE: {
      struct ClearTimer {
        Clock::time_point t0 = Clock::now();
        ~ClearTimer() {
          if (prof::enabled())
            prof::record_call(10007, 0, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()),
                              "rsx:clear");
        }
      } clear_timer;
      if (GpuRenderer* g = gpu()) {
        if (trace_flip_active(stats.flips.load(), false)) {
          K().logf("flip %llu clear: mask %02X fmt %08X clipH %08X clipV %08X cbase %08X p%u zbase %08X p%u",
                   (unsigned long long)stats.flips.load(), value, regs_[m::SURFACE_FORMAT], regs_[m::SURFACE_CLIP_H],
                   regs_[m::SURFACE_CLIP_V],
                   offset_to_address(regs_[m::SURFACE_COLOR_AOFFSET], regs_[m::CONTEXT_DMA_COLOR_A] == kDmaMain ? 1 : 0),
                   regs_[m::SURFACE_PITCH_A],
                   offset_to_address(regs_[m::SURFACE_ZETA_OFFSET], regs_[m::CONTEXT_DMA_ZETA] == kDmaMain ? 1 : 0),
                   regs_[m::SURFACE_PITCH_Z]);
        }
        if (g->clear(draw_env(), value)) {
          ++stats.clears;
          break;
        }
      }
      // Fallback: clears run on the CPU against guest memory: bring it up to date first,
      // then have the GPU reload whatever was cleared.
      sync(true);
      clear_surface(value);
      if (value & 3) clear_depth(draw_env(), value);
      if (gpu_) {
        uint32_t aa_x = 1, aa_y = 1;
        surface_aa_scale(regs_[m::SURFACE_FORMAT], aa_x, aa_y);
        const uint32_t h = ((regs_[m::SURFACE_CLIP_V] & 0xFFFF) + (regs_[m::SURFACE_CLIP_V] >> 16)) * aa_y;
        const uint32_t ca = offset_to_address(regs_[m::SURFACE_COLOR_AOFFSET], regs_[m::CONTEXT_DMA_COLOR_A] == 0xFEED0001 ? 1 : 0);
        const uint32_t za = offset_to_address(regs_[m::SURFACE_ZETA_OFFSET], regs_[m::CONTEXT_DMA_ZETA] == 0xFEED0001 ? 1 : 0);
        if ((value & 0xF0) && ca) gpu_->invalidate(ca, regs_[m::SURFACE_PITCH_A] * h);
        if ((value & 0x03) && za) gpu_->invalidate(za, regs_[m::SURFACE_PITCH_Z] * h);
      }
      break;
    }
    case M_BEGIN_END:
      if (value) {
        pending_ = DrawCall{value, false, {}};
        in_draw_ = true;
      } else if (in_draw_) {
        in_draw_ = false;
        const auto t0 = Clock::now();
        end_draw();
        if (prof::enabled())
          prof::record_call(10006, 0, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()),
                            "rsx:draw");
      }
      break;
    case m::DRAW_ARRAYS * 4:
      if (in_draw_) pending_.ranges.push_back({value & 0xFFFFFF, (value >> 24) + 1});
      break;
    case m::DRAW_INDEX_ARRAY * 4:
      if (in_draw_) {
        pending_.indexed = true;
        pending_.ranges.push_back({value & 0xFFFFFF, (value >> 24) + 1});
      }
      break;
    case m::TRANSFORM_PROGRAM_LOAD * 4:
      vp_load_ = value;
      vp_word_count_ = 0;
      break;
    case m::TRANSFORM_PROGRAM_START * 4:
      vp_.start = value % 512;
      ++vp_.prog_ver;
      break;
    case m::TRANSFORM_CONSTANT_LOAD * 4:
      const_load_ = value;
      const_word_ = 0;
      break;
    case kMethodFlip: {
      const auto t0 = Clock::now();
      flip(value);
      if (prof::enabled())
        prof::record_call(10005, 0, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()),
                          "rsx:flip");
      if (trace && stats.flips.load() <= 12)
        k.logf("rsx trace: flip %u took %.1f ms", value, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
      break;
    }
    case kMethodWaitLabel + 4: {
      const uint32_t index = regs_[kMethodWaitLabel / 4];
      const auto t0 = Clock::now();
      for (uint32_t spin = 0; !stop_ && k.r32(label_addr(index & 0xFF)) != value; ++spin)
        spin < 4000 ? std::this_thread::yield() : std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (trace)
        k.logf("rsx trace: waited %.1f ms for label %u == 0x%X", std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
               index & 0xFF, value);
      if (prof::enabled())
        prof::record_call(10004, index & 0xFF,
                          uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()), "rsx:waitlabel");
      break;
    }
    default:
      // Vertex program upload window: 4 words per instruction, slot auto-increments.
      if (method >= m::TRANSFORM_PROGRAM * 4 && method < (m::TRANSFORM_PROGRAM + 32) * 4) {
        vp_words_[vp_word_count_++] = value;
        if (vp_word_count_ == 4) {
          vp_.program[vp_load_ % 512] = decode_vp(vp_words_);
          vp_load_++;
          vp_word_count_ = 0;
          ++vp_.prog_ver;
        }
      } else if (method >= m::TRANSFORM_CONSTANT * 4 && method < (m::TRANSFORM_CONSTANT + 32) * 4) {
        vp_.constants[const_load_ % 512][const_word_] = std::bit_cast<float>(value);
        ++vp_.const_ver;
        if (++const_word_ == 4) {
          const_word_ = 0;
          const_load_++;
        }
      } else if (method >= 0x1C00 && method < 0x1D00) {  // NV4097_SET_VERTEX_DATA4F_M
        const uint32_t idx = (method - 0x1C00) / 16;
        const uint32_t comp = ((method - 0x1C00) / 4) & 3;
        const_attribs_[idx][comp] = std::bit_cast<float>(value);
      } else if (method >= 0x1880 && method < 0x1900) {  // NV4097_SET_VERTEX_DATA2F_M
        const uint32_t idx = (method - 0x1880) / 8;
        const uint32_t comp = ((method - 0x1880) / 4) & 1;
        if (comp == 0) const_attribs_[idx] = {std::bit_cast<float>(value), 0, 0, 1};
        else const_attribs_[idx][1] = std::bit_cast<float>(value);
      } else if (method >= 0x1940 && method < 0x1980) {  // NV4097_SET_VERTEX_DATA4UB_M
        const uint32_t idx = (method - 0x1940) / 4;
        const_attribs_[idx] = {(value & 0xFF) / 255.0f, ((value >> 8) & 0xFF) / 255.0f,
                               ((value >> 16) & 0xFF) / 255.0f, ((value >> 24) & 0xFF) / 255.0f};
      } else if (method >= 0x1980 && method < 0x1A00) {  // NV4097_SET_VERTEX_DATA4S_M
        const uint32_t idx = (method - 0x1980) / 8;
        const uint32_t pair = ((method - 0x1980) / 4) & 1;
        const_attribs_[idx][2 * pair] = float(int16_t(value & 0xFFFF));
        const_attribs_[idx][2 * pair + 1] = float(int16_t(value >> 16));
      } else if (method >= 0x1900 && method < 0x1940) {  // NV4097_SET_VERTEX_DATA2S_M
        const uint32_t idx = (method - 0x1900) / 4;
        const_attribs_[idx] = {float(int16_t(value & 0xFFFF)), float(int16_t(value >> 16)), 0, 1};
      }
      break;
  }
}

DrawEnv Rsx::draw_env() {
  DrawEnv env;
  env.mem = K().base();
  env.regs = regs_.data();
  env.vp = &vp_;
  env.const_attribs = const_attribs_;
  env.to_address = [this](uint32_t offset, uint32_t location) { return offset_to_address(offset, location); };
  return env;
}

void Rsx::end_draw() {
  ++stats.draws;
  const DrawEnv env = draw_env();
  auto report = [&](const std::string& what) {
    if (std::find(reported_.begin(), reported_.end(), what) == reported_.end()) {
      reported_.push_back(what);
      K().logf("rsx: %s", what.c_str());
    }
  };
  PreparedDraw& d = prepared_draw_;
  std::string err;
  static const bool trace = std::getenv("PS3RECOMP_RSX_TRACE") != nullptr;
  static const bool trace_flip_env = std::getenv("PS3RECOMP_TRACE_FLIP") != nullptr;
  static const bool cpu_vp = std::getenv("PS3RECOMP_CPU_VP") != nullptr;
  if (!cpu_vp && !trace_flip_env && !(trace && stats.draws.load() <= 3) &&
      !regs_[M_ZPASS_PIXEL_COUNT_ENABLE / 4]) {
    if (GpuRenderer* g = gpu()) {
      if (!prepare_draw_state(env, d, &err)) {
        ++stats.skipped_draws;
        report("draw skipped: " + err);
        return;
      }
      uint32_t drawn_tris = 0;
      if (g->draw_raw(env, pending_, d, drawn_tris, &err)) {
        stats.triangles += drawn_tris;
        ++gpu_draws;
        return;
      }
    }
  }
  if (!prepare_draw(env, pending_, d, &err)) {
    ++stats.skipped_draws;
    report("draw skipped: " + err);
    return;
  }
  stats.triangles += d.tris.size() / 3;
  if (regs_[M_ZPASS_PIXEL_COUNT_ENABLE / 4]) {
    float zx0 = 1e9f, zx1 = -1e9f, zy0 = 1e9f, zy1 = -1e9f;
    for (const PreparedVertex& v : d.tris) {
      zx0 = std::min(zx0, v.x); zx1 = std::max(zx1, v.x);
      zy0 = std::min(zy0, v.y); zy1 = std::max(zy1, v.y);
    }
    uint32_t area = 64u;
    if (!d.tris.empty() && zx1 > zx0 && zy1 > zy0) {
      const float cx0 = std::max(zx0, float(d.min_x)), cx1 = std::min(zx1, float(d.max_x));
      const float cy0 = std::max(zy0, float(d.min_y)), cy1 = std::min(zy1, float(d.max_y));
      if (cx1 > cx0 && cy1 > cy0) area = std::max<uint32_t>(64u, uint32_t((cx1 - cx0) * (cy1 - cy0)));
    }
    zpass_pixels_ += area;
  }
  const uint64_t cur_flip = stats.flips.load();
  if (trace_flip_active(cur_flip, d.depth_test && d.depth_write && d.surface_w >= 512 && d.tris.size() >= 30)) {
    std::string tx;
    for (int i = 0; i < 16; ++i) {
      const TexUnit& t = d.tex[i];
      if (!t.enabled) continue;
      const uint32_t* tr = regs_.data() + m::TEXTURE_BASE + 8 * i;
      char b[160];
      std::snprintf(b, sizeof b, " t%d=%08X f%02X(a%08X c%08X r%08X flt%08X) %ux%u p%u%s", i, t.address, t.format,
                    tr[2], tr[3], tr[4], tr[5], t.width, t.height, t.pitch, t.linear_layout ? "L" : "S");
      tx += b;
    }
    float vx0 = 1e9f, vx1 = -1e9f, vy0 = 1e9f, vy1 = -1e9f;
    for (const PreparedVertex& v : d.tris) {
      vx0 = std::min(vx0, v.x); vx1 = std::max(vx1, v.x);
      vy0 = std::min(vy0, v.y); vy1 = std::max(vy1, v.y);
    }
    auto fb = [](uint32_t v) { return std::bit_cast<float>(v); };
    K().logf("flip %llu draw %llu: rt %08X p%u surf %ux%u bounds x%d-%d y%d-%d fmt %08X clipH %08X scH %08X z %08X p%u dfmt %u dt%d dw%d df%X b=%d(%X/%X,%X/%X,%X/%X bc=%08X,%08X) cm=%08X at=%d(%02X,%X) cu=%d/%X/%X vpS %.1f,%.1f vpO %.1f,%.1f tris %zu verts x%.0f-%.0f y%.0f-%.0f fp %016llX",
             (unsigned long long)cur_flip, (unsigned long long)stats.draws.load(), d.color_base, d.pitch, d.surface_w,
             d.surface_h, d.min_x, d.max_x, d.min_y, d.max_y, regs_[m::SURFACE_FORMAT], regs_[m::SURFACE_CLIP_H],
             regs_[m::SCISSOR_H], d.zbase, d.zpitch, d.depth_fmt, d.depth_test ? 1 : 0, d.depth_write ? 1 : 0,
             d.depth_func, d.blend ? 1 : 0, d.sf_rgb, d.sf_a, d.df_rgb, d.df_a, d.eq_rgb, d.eq_a,
             regs_[m::BLEND_COLOR], regs_[m::BLEND_COLOR2], d.color_mask, d.alpha_test ? 1 : 0, d.alpha_ref,
             d.alpha_func, regs_[m::CULL_FACE_ENABLE] ? 1 : 0, regs_[m::CULL_FACE], regs_[m::FRONT_FACE],
             fb(regs_[m::VIEWPORT_SCALE]), fb(regs_[m::VIEWPORT_SCALE + 1]), fb(regs_[m::VIEWPORT_OFFSET]),
             fb(regs_[m::VIEWPORT_OFFSET + 1]), d.tris.size() / 3, vx0, vx1, vy0, vy1,
             (unsigned long long)d.fp_hash);
    if (!tx.empty()) K().logf("  tex:%s", tx.c_str());
    if (d.tris.empty()) {
      const auto& vc = vp_.constants;
      static const vec4 z4{};
      const vec4& c256 = vc.size() > 256 ? vc[256] : z4;
      const vec4& c257 = vc.size() > 257 ? vc[257] : z4;
      const vec4& c258 = vc.size() > 258 ? vc[258] : z4;
      const vec4& c259 = vc.size() > 259 ? vc[259] : z4;
      const vec4& c260 = vc.size() > 260 ? vc[260] : z4;
      const vec4& c261 = vc.size() > 261 ? vc[261] : z4;
      K().logf("  tris0 prim=%u ranges=%zu raw=%u nf=%u cl=%u cu=%u tri0[(%.3g,%.3g,%.3g,%.3g),(%.3g,%.3g,%.3g,%.3g),(%.3g,%.3g,%.3g,%.3g)] vp_start=%zu c256=[(%.3g,%.3g,%.3g,%.3g),(%.3g,%.3g,%.3g,%.3g),(%.3g,%.3g,%.3g,%.3g),(%.3g,%.3g,%.3g,%.3g)] c260=[(%.3g,%.3g,%.3g,%.3g),(%.3g,%.3g,%.3g,%.3g)]",
               pending_.primitive, pending_.ranges.size(), d.raw_tris, d.rej_nonfinite, d.rej_clip, d.rej_cull,
               d.first_clip[0][0], d.first_clip[0][1], d.first_clip[0][2], d.first_clip[0][3],
               d.first_clip[1][0], d.first_clip[1][1], d.first_clip[1][2], d.first_clip[1][3],
               d.first_clip[2][0], d.first_clip[2][1], d.first_clip[2][2], d.first_clip[2][3], vp_.start,
               c256[0], c256[1], c256[2], c256[3], c257[0], c257[1], c257[2], c257[3],
               c258[0], c258[1], c258[2], c258[3], c259[0], c259[1], c259[2], c259[3],
               c260[0], c260[1], c260[2], c260[3], c261[0], c261[1], c261[2], c261[3]);
    }
    static std::vector<uint64_t> dumped_fp;
    if (std::find(dumped_fp.begin(), dumped_fp.end(), d.fp_hash) == dumped_fp.end()) {
      dumped_fp.push_back(d.fp_hash);
      const uint32_t sp = regs_[m::SHADER_PROGRAM];
      const uint32_t fp_addr = offset_to_address(sp & ~3u, (sp & 3) == 2 ? 1 : 0);
      auto fp_w = [&](size_t byte_off) -> uint32_t {
        if (!fp_addr) return 0;
        const uint8_t* p = K().base() + fp_addr + byte_off;
        const uint32_t be = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
        return (be << 16) | (be >> 16);
      };
      K().logf("rsx_fp %016llX (%zu insns, fp32=%d, addr=%08X):", (unsigned long long)d.fp_hash, d.fp.size(),
               d.fp32_out ? 1 : 0, fp_addr);
      size_t boff = 0;
      for (size_t fi = 0; fi < d.fp.size(); ++fi) {
        const FpInsn& f = d.fp[fi];
        const uint32_t w0 = fp_w(boff), w1 = fp_w(boff + 4), w2 = fp_w(boff + 8), w3 = fp_w(boff + 12);
        boff += 16;
        if (f.has_constant) boff += 16;
        K().logf("  fp[%zu] %08X %08X %08X %08X | %s", fi, w0, w1, w2, w3, disasm_fp(f).c_str());
      }
      for (size_t k = 0; k < 32 && vp_.start + k < vp_.program.size(); ++k) {
        const VpInsn& vi = vp_.program[vp_.start + k];
        K().logf("  vp[%zu] %s", k, disasm_vp(vi).c_str());
        if (vi.end) break;
      }
    }
  }
  if (trace && stats.draws.load() <= 3) {
    K().logf("rsx trace: draw prim %u, %zu range(s), %zu triangle(s) after clip/cull; target 0x%08X pitch %u, %dx%d",
             pending_.primitive, pending_.ranges.size(), d.tris.size() / 3, d.color_base, d.pitch, d.max_x, d.max_y);
    for (size_t i = 0; i < std::min<size_t>(d.tris.size(), 3); ++i) {
      const PreparedVertex& v = d.tris[i];
      K().logf("rsx trace:   v%zu window (%.1f, %.1f, %.3f) 1/w %.3f col0 (%.2f %.2f %.2f %.2f)", i, v.x, v.y, v.z,
               v.inv_w, v.vary[0][0], v.vary[0][1], v.vary[0][2], v.vary[0][3]);
    }
    for (size_t k = 0; k < std::min<size_t>(vp_.program.size(), 8); ++k)
      K().logf("rsx trace:   vp[%zu] %s", k, disasm_vp(vp_.program[vp_.start + k]).c_str());
    for (const FpInsn& f : d.fp) K().logf("rsx trace:   fp %s", disasm_fp(f).c_str());
  }

  if (GpuRenderer* g = gpu()) {
    if (g->draw(env, d, &err)) {
      ++gpu_draws;
      return;
    }
    // GPU path can't do this one: bring guest memory up to date, draw on the CPU,
    // and make the GPU reload the surfaces the CPU just changed.
    report("draw on CPU (GPU path: " + err + ")");
    g->flush(env.mem);
  }
  DrawStats ds;
  if (!raster_cpu(env, d, ds, &err)) {
    ++stats.skipped_draws;
    report("draw skipped: " + err);
  }
  ++cpu_draws;
  stats.pixels += ds.pixels;
  if (gpu_) {
    gpu_->invalidate(d.color_base, d.pitch * d.surface_h);
    if (d.zbase) gpu_->invalidate(d.zbase, d.zpitch * d.surface_h);
  }
}

namespace {

uint32_t sfm_bpp(uint32_t fmt) {
  switch (fmt) {
    case 1: case 2: case 7: case 11: return 2;
    case 3: case 4: return 4;
    case 16: return 1;
    default: return 0;
  }
}

uint32_t surf2d_bpp(uint32_t fmt) {
  switch (fmt) {
    case 1: return 1;
    case 4: case 7: return 2;
    case 10: case 11: return 4;
    default: return 0;
  }
}

uint32_t unpack_sfm_argb(const uint8_t* mem, uint32_t ea, uint32_t fmt) {
  switch (fmt) {
    case 3: return ppu::ld32(mem, ea);
    case 4: return ppu::ld32(mem, ea) | 0xFF000000u;
    case 1: {
      const uint16_t p = ppu::ld16(mem, ea);
      const uint32_t a = (p >> 15) ? 0xFFu : 0u;
      const uint32_t r = ((p >> 10) & 0x1F) * 255 / 31;
      const uint32_t g = ((p >> 5) & 0x1F) * 255 / 31;
      const uint32_t b = (p & 0x1F) * 255 / 31;
      return (a << 24) | (r << 16) | (g << 8) | b;
    }
    case 2: {
      const uint16_t p = ppu::ld16(mem, ea);
      const uint32_t r = ((p >> 10) & 0x1F) * 255 / 31;
      const uint32_t g = ((p >> 5) & 0x1F) * 255 / 31;
      const uint32_t b = (p & 0x1F) * 255 / 31;
      return 0xFF000000u | (r << 16) | (g << 8) | b;
    }
    case 7: {
      const uint16_t p = ppu::ld16(mem, ea);
      const uint32_t r = ((p >> 11) & 0x1F) * 255 / 31;
      const uint32_t g = ((p >> 5) & 0x3F) * 255 / 63;
      const uint32_t b = (p & 0x1F) * 255 / 31;
      return 0xFF000000u | (r << 16) | (g << 8) | b;
    }
    case 16: {
      const uint32_t y = mem[ea];
      return 0xFF000000u | (y << 16) | (y << 8) | y;
    }
    default: return 0;
  }
}

void store_surf2d_argb(uint8_t* mem, uint32_t ea, uint32_t fmt, uint32_t argb) {
  switch (fmt) {
    case 10: case 11: ppu::st32(mem, ea, argb); break;
    case 4: {
      const uint16_t p = uint16_t((((argb >> 19) & 0x1F) << 11) | (((argb >> 10) & 0x3F) << 5) | ((argb >> 3) & 0x1F));
      ppu::st16(mem, ea, p);
      break;
    }
    case 1: mem[ea] = uint8_t(argb & 0xFF); break;
    default: break;
  }
}

uint32_t blend_argb(uint32_t src, uint32_t dst, uint32_t op) {
  if (op == 0 || op == 3) return src;
  const uint32_t a = src >> 24;
  if (op == 4) {  // SRCCOPY_PREMULT
    auto pm = [&](int s) { return (((src >> s) & 0xFF) * a) / 255u; };
    return (a << 24) | (pm(16) << 16) | (pm(8) << 8) | pm(0);
  }
  const uint32_t inv = 255u - a;
  auto ch = [&](int s) {
    const uint32_t sv = (src >> s) & 0xFF, dv = (dst >> s) & 0xFF;
    const uint32_t out = (op == 5) ? (sv + (dv * inv) / 255u) : ((sv * a + dv * inv) / 255u);
    return out > 255u ? 255u : out;
  };
  return (ch(24) << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0);
}

}  // namespace

static void execute_nv3089_blit(const std::vector<uint32_t>& regs, lv2::Kernel& k, GpuRenderer* gpu_ptr,
                                const std::function<uint32_t(uint32_t, uint32_t)>& to_addr,
                                const std::function<void(const std::string&)>& report_fn) {
  // Subchannel 6 = NV3089 (Scaled Image From Memory), target surface on subchannel 3 = NV3062.
  constexpr uint32_t kSfm = 0x3000, k2d = 0x1800;
  const uint32_t ctx = regs[kSfm + 0x198 / 4];
  if (ctx && ctx != 0x313371C3u) {
    report_fn("nv3089 unsupported context surface");
    return;
  }
  const uint32_t src_loc = (regs[kSfm + 0x184 / 4] == 0xFEED0001u) ? 1u : 0u;
  const uint32_t dst_loc = (regs[k2d + 0x188 / 4] == 0xFEED0001u) ? 1u : 0u;
  const uint32_t in_fmt = regs[kSfm + 0x300 / 4];
  const uint32_t op = regs[kSfm + 0x304 / 4];
  const uint32_t dst_fmt = regs[k2d + 0x300 / 4];
  const uint32_t sbpp = sfm_bpp(in_fmt), dbpp = surf2d_bpp(dst_fmt);
  if (!sbpp || !dbpp) {
    report_fn("nv3089 unsupported format");
    return;
  }
  const int32_t clip_x = int16_t(regs[kSfm + 0x308 / 4] & 0xFFFF);
  const int32_t clip_y = int16_t(regs[kSfm + 0x308 / 4] >> 16);
  const int32_t clip_w = int32_t(regs[kSfm + 0x30C / 4] & 0xFFFF);
  const int32_t clip_h = int32_t(regs[kSfm + 0x30C / 4] >> 16);
  const int32_t out_x = int16_t(regs[kSfm + 0x310 / 4] & 0xFFFF);
  const int32_t out_y = int16_t(regs[kSfm + 0x310 / 4] >> 16);
  const int32_t out_w = int32_t(regs[kSfm + 0x314 / 4] & 0xFFFF);
  const int32_t out_h = int32_t(regs[kSfm + 0x314 / 4] >> 16);
  const int32_t in_w = int32_t(regs[kSfm + 0x400 / 4] & 0xFFFF);
  const int32_t in_h = int32_t(regs[kSfm + 0x400 / 4] >> 16);
  const uint32_t src_pitch = regs[kSfm + 0x404 / 4] & 0xFFFFu;
  const uint32_t dst_pitch = (regs[k2d + 0x304 / 4] >> 16) & 0xFFFFu;
  const uint32_t src_off = regs[kSfm + 0x408 / 4];
  const uint32_t dst_off = regs[k2d + 0x30C / 4];
  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0 || clip_w <= 0 || clip_h <= 0 || !dst_pitch || !src_pitch)
    return;

  const int32_t x0 = std::max<int32_t>({0, clip_x, out_x});
  const int32_t y0 = std::max<int32_t>({0, clip_y, out_y});
  const int32_t x1 = std::min<int32_t>(clip_x + clip_w, out_x + out_w);
  const int32_t y1 = std::min<int32_t>(clip_y + clip_h, out_y + out_h);
  if (x0 >= x1 || y0 >= y1) return;

  if (gpu_ptr && src_loc == 0) {
    if (const uint32_t src_base = to_addr(src_off, src_loc))
      gpu_ptr->flush_range(k.base(), src_base, uint32_t(in_h) * src_pitch);
  }
  if (gpu_ptr && dst_loc == 0) {
    if (const uint32_t dst_base = to_addr(dst_off + uint32_t(y0) * dst_pitch, dst_loc))
      gpu_ptr->flush_range(k.base(), dst_base, uint32_t(y1 - y0) * dst_pitch);
  }
  uint8_t* mem = k.base();
  const int32_t ds_dx = int32_t(regs[kSfm + 0x318 / 4]);
  const int32_t dt_dy = int32_t(regs[kSfm + 0x31C / 4]);
  const uint32_t in_pt = regs[kSfm + 0x40C / 4];
  const int32_t u0_4 = int16_t(in_pt & 0xFFFF);
  const int32_t v0_4 = int16_t(in_pt >> 16);
  const bool raw_1to1 = (op == 0 || op == 3) && ds_dx == 0x100000 && dt_dy == 0x100000 && sbpp == dbpp &&
                        (u0_4 & 0xF) == 0 && (v0_4 & 0xF) == 0 &&
                        !((in_fmt == 3 || in_fmt == 4) && dst_fmt != 10 && dst_fmt != 11);

  if (raw_1to1) {
    const int32_t sx0 = std::clamp((u0_4 >> 4) + (x0 - out_x), 0, in_w - 1);
    const int32_t copy_w = std::min(x1 - x0, in_w - sx0);
    const int32_t rows = y1 - y0;
    const bool rev_y = (src_loc == dst_loc && dst_off > src_off);
    for (int32_t ry = 0; ry < rows; ++ry) {
      const int32_t dy = rev_y ? (y1 - 1 - ry) : (y0 + ry);
      const int32_t sy = std::clamp((v0_4 >> 4) + (dy - out_y), 0, in_h - 1);
      const uint32_t sea = to_addr(src_off + uint32_t(sy) * src_pitch + uint32_t(sx0) * sbpp, src_loc);
      const uint32_t dea = to_addr(dst_off + uint32_t(dy) * dst_pitch + uint32_t(x0) * dbpp, dst_loc);
      if (sea && dea) std::memmove(mem + dea, mem + sea, size_t(copy_w) * dbpp);
    }
  } else {
    const uint32_t dst_as_sfm = (dst_fmt == 4) ? 7u : (dst_fmt == 1 ? 16u : 3u);
    const int64_t u0_20 = int64_t(u0_4) << 16;
    const int64_t v0_20 = int64_t(v0_4) << 16;
    for (int32_t dy = y0; dy < y1; ++dy) {
      const int32_t sy = std::clamp(int32_t((v0_20 + int64_t(dy - out_y) * dt_dy) >> 20), 0, in_h - 1);
      const uint32_t srow = to_addr(src_off + uint32_t(sy) * src_pitch, src_loc);
      const uint32_t drow = to_addr(dst_off + uint32_t(dy) * dst_pitch, dst_loc);
      if (!srow || !drow) continue;
      for (int32_t dx = x0; dx < x1; ++dx) {
        const int32_t sx = std::clamp(int32_t((u0_20 + int64_t(dx - out_x) * ds_dx) >> 20), 0, in_w - 1);
        const uint32_t src_px = unpack_sfm_argb(mem, srow + uint32_t(sx) * sbpp, in_fmt);
        const uint32_t dst_px =
            (op == 0 || op == 3 || op == 4) ? 0u : unpack_sfm_argb(mem, drow + uint32_t(dx) * dbpp, dst_as_sfm);
        store_surf2d_argb(mem, drow + uint32_t(dx) * dbpp, dst_fmt, blend_argb(src_px, dst_px, op));
      }
    }
  }

  if (gpu_ptr && dst_loc == 0) {
    if (const uint32_t base_ea = to_addr(dst_off + uint32_t(y0) * dst_pitch, dst_loc))
      gpu_ptr->invalidate(base_ea, uint32_t(y1 - y0) * dst_pitch);
  }
}

void Rsx::run() {
  lv2::Kernel& k = K();
  bool warned = false;
  uint32_t idle = 0;
  Clock::time_point idle_since_{};
  while (!stop_) {
    const uint32_t put = k.r32(control_ + 0);
    uint32_t get = k.r32(control_ + 4);
    if (put == get) {
      // The game often waits on us right after kicking work, so stay responsive:
      // yield for a while before falling back to (1 ms resolution) sleeps.
      if (idle == 0 && prof::enabled()) idle_since_ = Clock::now();
      poll_gpu();  // finish (and present) frames the GPU has completed
      if (++idle < 4000) std::this_thread::yield();
      else std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (idle && prof::enabled())
      prof::record_call(10000, 0, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - idle_since_).count()),
                        "rsx:fifo-empty");
    idle = 0;
    uint32_t ea = 0;
    if (!io_to_ea(get, &ea)) {
      if (!warned) k.logf("rsx: FIFO get 0x%08X is not mapped", get);
      warned = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    const uint32_t cmd = ppu::ld32(k.base(), ea);
    if ((cmd & kOldJumpMask) == kOldJump) {
      get = cmd & 0x1FFFFFFC;
    } else if ((cmd & 3) == 1) {
      get = cmd & ~3u;
    } else if ((cmd & 3) == 2) {
      call_stack_.push_back(get + 4);
      get = cmd & ~3u;
    } else if (cmd == kReturn) {
      if (call_stack_.empty()) {
        k.logf("rsx: FIFO return with empty call stack");
        get += 4;
      } else {
        get = call_stack_.back();
        call_stack_.pop_back();
      }
    } else {
      const uint32_t count = (cmd >> 18) & 0x7FF;
      const uint32_t method = cmd & 0xFFFC;
      const bool same = (cmd & kNonIncrement) != 0;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t m = same ? method : method + 4 * i;
        const uint32_t v = read_io32(get + 4 + 4 * i);
        auto warn_once = [this, &k](const std::string& msg) {
          if (std::find(reported_.begin(), reported_.end(), msg) == reported_.end()) {
            reported_.push_back(msg);
            k.logf("rsx: %s", msg.c_str());
          }
        };
        execute(m, v);
        if (m == 0xC40C) {
          execute_nv3089_blit(regs_, k, gpu(),
                              [this](uint32_t o, uint32_t l) { return offset_to_address(o, l); }, warn_once);
        } else if (m < 0x10000 && (m & 0xE000) != 0 && (m & 0x1FFC) == 0x032C) {
          warn_once("unhandled nv0039 buffer notify");
        }
      }
      get += 4 * (count + 1);
      if ((++stats.commands & 255) == 0) poll_gpu();
    }
    k.w32(control_ + 4, get);
  }
}

void Rsx::vblank_loop() {
  // 59.94 Hz (high-resolution timer on Windows so VBlank doesn't quantize to 15.6 ms ticks).
#if defined(_WIN32)
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
  HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /*HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
  if (!timer) timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
#endif
  const auto period = std::chrono::microseconds(16683);
  auto next = Clock::now() + period;
  static const bool trace = std::getenv("PS3RECOMP_RSX_TRACE") != nullptr;
  while (!stop_) {
    const auto now = Clock::now();
    if (next > now) {
#if defined(_WIN32)
      const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(next - now);
      LARGE_INTEGER due;
      due.QuadPart = -int64_t(std::max<int64_t>(1, d.count() / 100));
      if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE))
        WaitForSingleObject(timer, INFINITE);
      else
        std::this_thread::sleep_until(next);
#else
      std::this_thread::sleep_until(next);
#endif
    } else if (now > next + std::chrono::milliseconds(50)) {
      next = now;
    }
    next += period;
    {
      std::lock_guard<std::mutex> lk(vblank_m_);
      ++stats.vblanks;
    }
    vblank_cv_.notify_all();
    if (trace && stats.vblanks % 60 == 0)
      K().logf("rsx trace: %llu commands, %llu draws (%llu skipped), %llu flips, %llu clears, put 0x%X get 0x%X",
               (unsigned long long)stats.commands.load(), (unsigned long long)stats.draws.load(),
               (unsigned long long)stats.skipped_draws.load(), (unsigned long long)stats.flips.load(),
               (unsigned long long)stats.clears.load(), K().r32(control_), K().r32(control_ + 4));
    call_handler(vblank_handler, 1);
  }
#if defined(_WIN32)
  if (timer) CloseHandle(timer);
#endif
}

}  // namespace rsx
