#pragma once
// Puts finished frames on screen. The RSX emulation hands over 32-bit BGRA pixels
// (host byte order: B, G, R, A) after each flip.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

// F4 in the game window: asks the renderer for a diagnostic capture (render targets + draw list).
extern std::atomic<int> g_capture_requests;
// F2 / F3 debug toggles (press counters, read by the renderer): shadow-sampler border mode, depth clamp.
extern std::atomic<int> g_debug_f2, g_debug_f3;
// F5 / F6: neutralise the hardware depth-cascade shadows / the VSM atlas, to find which one darkens a scene.
extern std::atomic<int> g_debug_f5, g_debug_f6;
// F7 / F8 / F9: draw stepper (off / render fewer scene draws / render more). F8 and F9 auto-repeat.
extern std::atomic<int> g_debug_f7, g_debug_f8, g_debug_f9;
// F11: automatic draw-stepper sweep that saves the scene after each step (stand still, ~10 s).
extern std::atomic<int> g_debug_f11;
// Microseconds the RSX thread has spent blocked on semaphore acquires (= waiting for the game's PPU/SPU work). Diagnostics.
extern std::atomic<uint64_t> g_rsx_wait_us;
// Microseconds the RSX thread has spent waiting for the emulated VBlank at flips. Diagnostics.
extern std::atomic<uint64_t> g_rsx_vblank_wait_us;
// Microseconds the RSX thread has spent inside Presenter::present (total, and the longest single call). Diagnostics.
extern std::atomic<uint64_t> g_rsx_present_us, g_rsx_present_max_us;
// F12: cycle the luminance value the CPU reads from the auto-exposure targets (real/zero/x0.25/x4/x16).
extern std::atomic<int> g_debug_f12;

class Presenter {
 public:
  virtual ~Presenter() = default;
  // Blocks until the frame is queued for display (vsync-paced on real displays).
  virtual void present(const uint32_t* bgra, uint32_t width, uint32_t height, uint32_t pitch_pixels) = 0;
  // False once the user closed the window.
  virtual bool alive() const { return true; }
  // Name of the GPU in use, or "" for non-GPU presenters.
  virtual std::string device_name() const { return {}; }
  // Live display settings update (resolution, 0=windowed/1=borderless, vsync 0..2, linear 0/1, fps title).
  virtual void configure(uint32_t /*w*/, uint32_t /*h*/, int /*display_mode*/, int /*vsync*/,
                         int /*upscale_filter*/, bool /*fps_counter*/) {}
  // Optional callbacks for in-game F1 settings hotkey and window focus changes.
  virtual void set_callbacks(std::function<void()> /*on_settings*/, std::function<void(bool)> /*on_focus*/) {}
};

// Vulkan presenter with its own window. Picks a discrete GPU when there is one.
// Returns nullptr (and logs why) when Vulkan or a window is unavailable.
std::unique_ptr<Presenter> create_vulkan_presenter(const std::string& title,
                                                   std::function<void(std::string_view)> log,
                                                   uint32_t width = 1280, uint32_t height = 720,
                                                   const std::string& icon_path = {});

// Keeps the last frame in memory (tests, headless runs).
class CapturePresenter : public Presenter {
 public:
  void present(const uint32_t* bgra, uint32_t w, uint32_t h, uint32_t pitch) override {
    width = w;
    height = h;
    pixels.assign(size_t(w) * h, 0);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x) pixels[size_t(y) * w + x] = bgra[size_t(y) * pitch + x];
    ++frames;
  }
  uint32_t width = 0, height = 0, frames = 0;
  std::vector<uint32_t> pixels;
};

}  // namespace gfx
