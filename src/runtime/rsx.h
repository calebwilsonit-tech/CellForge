#pragma once
// RSX (the PS3 GPU) as seen by a game through libgcm:
//   * 256 MB of local (video) memory at guest 0xC0000000,
//   * main memory mapped into the RSX "IO" address space in 1 MB pages,
//   * a command FIFO in IO memory driven by the put/get/ref control registers,
//   * labels (semaphores), reports, display buffers and flips.
//
// A puller thread executes the FIFO. Phase 1 handles the command stream itself
// (jumps, calls, references, semaphores), surface state, colour clears and flips.
// 3D draw calls are counted but not rendered yet.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <functional>
#include <thread>
#include <vector>

#include "presenter.h"
#include "rsx_draw.h"
#include "rsx_program.h"

namespace rsx {

class GpuRenderer;

constexpr uint32_t kLocalBase = 0xC0000000;
constexpr uint32_t kLocalSize = 0x10000000;   // 256 MB reserved
constexpr uint32_t kLocalUsable = 0x0F900000; // what libgcm reports to games (249 MB)
constexpr uint32_t kReportsOffset = 0x0FE00000; // reports live at the top of local memory
constexpr uint32_t kIoPage = 0x100000;        // 1 MB IO pages
constexpr uint32_t kMaxIoPages = 512;         // 512 MB of IO space
constexpr uint32_t kEaPages = 3072;           // 3 GB of main memory in 1 MB pages
constexpr uint32_t kIoUnmapped = 0xFFFF;

// Private FIFO methods written by our cellGcm HLE (never used by real hardware).
constexpr uint32_t kMethodFlip = 0xFEAC;        // data: display buffer id
constexpr uint32_t kMethodWaitLabel = 0xFEB0;   // data: label index, value

struct DisplayBuffer {
  uint32_t offset = 0, pitch = 0, width = 0, height = 0;
  bool valid = false;
};

struct Stats {
  std::atomic<uint64_t> commands{0}, flips{0}, clears{0}, draws{0}, vblanks{0};
  std::atomic<uint64_t> triangles{0}, pixels{0}, skipped_draws{0};
};

class Rsx {
 public:
  static Rsx& get();

  // Called by _cellGcmInitBody. Maps [io_address, io_address + io_size) to IO offset 0.
  void init(uint32_t io_address, uint32_t io_size);
  bool initialized() const { return initialized_; }
  void shutdown();

  // ---- memory mapping ----
  bool map_io(uint32_t ea, uint32_t io, uint32_t size);   // explicit mapping
  bool map_main(uint32_t ea, uint32_t size, uint32_t* io_out);  // pick free IO pages
  bool unmap_ea(uint32_t ea);
  bool unmap_io(uint32_t io);
  bool ea_to_io(uint32_t ea, uint32_t* io) const;
  bool io_to_ea(uint32_t io, uint32_t* ea) const;
  // Guest address for an RSX offset in local (location 0) or main/IO (location 1) memory.
  uint32_t offset_to_address(uint32_t offset, uint32_t location) const;

  // Guest-visible tables (CellGcmOffsetTable): u16 io page per ea page, u16 ea page per io page.
  uint32_t io_table_addr() const { return io_table_; }
  uint32_t ea_table_addr() const { return ea_table_; }

  // ---- registers ----
  uint32_t control_addr() const { return control_; }    // { u32 put, u32 get, u32 ref }
  uint32_t label_addr(uint32_t index) const { return labels_ + 16 * index; }
  uint32_t report_addr(uint32_t index) const { return kLocalBase + kReportsOffset + 16 * index; }
  uint32_t io_address() const { return io_address_; }
  uint32_t io_size() const { return io_size_; }

  // ---- display ----
  void set_display_buffer(uint32_t id, uint32_t offset, uint32_t pitch, uint32_t width, uint32_t height);
  uint32_t display_info_addr() const { return display_info_; }
  void set_presenter(std::shared_ptr<gfx::Presenter> p);
  // Used by init() to create a presenter when none was set (e.g. open a window).
  std::function<std::shared_ptr<gfx::Presenter>()> presenter_factory;
  void flip(uint32_t id, bool immediate = false);  // also used by cellGcmSetFlipImmediate
  void present_display(const DisplayBuffer& d, uint32_t base, uint64_t flip_no);

  // Renderer choice: GPU (Vulkan) when available, else the CPU reference renderer.
  std::atomic<bool> use_gpu{true};
  // Writes all GPU-rendered surfaces back to guest memory.
  void sync(bool include_depth = true);
  // Completes GPU frames that have finished (non-blocking).
  void poll_gpu();
  std::string gpu_name();
  std::atomic<uint64_t> gpu_draws{0}, cpu_draws{0};
  std::atomic<uint32_t> flip_mode{2};       // 1 = CELL_GCM_DISPLAY_HSYNC, 2 = CELL_GCM_DISPLAY_VSYNC
  std::atomic<uint32_t> flip_status{0};     // 0 = flip done, 1 = waiting
  std::atomic<uint32_t> flip_handler{0};    // guest OPD, called after each flip
  std::atomic<uint32_t> vblank_handler{0};  // guest OPD, called every vblank
  std::atomic<uint64_t> last_flip_time{0};
  Stats stats;

 private:
  Rsx();
  ~Rsx();
  void run();
  void vblank_loop();
  void execute(uint32_t method, uint32_t value);
  void clear_surface(uint32_t mask);
  void call_handler(uint32_t opd, uint64_t arg);
  uint32_t read_io32(uint32_t io) const;

  bool initialized_ = false;
  uint32_t io_address_ = 0, io_size_ = 0;
  uint32_t control_ = 0, labels_ = 0, io_table_ = 0, ea_table_ = 0, display_info_ = 0;
  uint16_t io_of_ea_[kEaPages] = {};
  uint16_t ea_of_io_[kMaxIoPages] = {};
  mutable std::mutex map_m_;

  DisplayBuffer displays_[8];
  std::mutex present_m_;
  std::shared_ptr<gfx::Presenter> presenter_;
  std::vector<uint32_t> frame_;

  // Method register file (word index = method / 4) and decoded state.
  std::vector<uint32_t> regs_ = std::vector<uint32_t>(0x10000 / 4, 0);
  uint32_t semaphore_offset_406e_ = 0, semaphore_offset_4097_ = 0;
  std::vector<uint32_t> call_stack_;

  // 3D state
  VpState vp_;
  vec4 const_attribs_[16] = {
      {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1},
      {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}};
  uint32_t vp_load_ = 0, vp_word_count_ = 0, vp_words_[4] = {};
  uint32_t const_load_ = 0, const_word_ = 0;
  uint32_t zpass_pixels_ = 0;
  DrawCall pending_;
  PreparedDraw prepared_draw_;
  bool in_draw_ = false;
  std::vector<std::string> reported_;  // draw problems already logged once
  void end_draw();
  DrawEnv draw_env();
  std::unique_ptr<class GpuRenderer> gpu_;
  bool gpu_tried_ = false;
  std::mutex gpu_m_;
  class GpuRenderer* gpu();

  std::atomic<bool> stop_{false};
  std::mutex vblank_m_;
  std::condition_variable vblank_cv_;
  uint64_t last_flip_vblank_ = 0;
  std::thread puller_, vblank_;
};

}  // namespace rsx
