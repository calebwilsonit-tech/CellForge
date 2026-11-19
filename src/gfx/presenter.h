#pragma once
// Puts finished frames on screen. The RSX emulation hands over 32-bit BGRA pixels
// (host byte order: B, G, R, A) after each flip.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

class Presenter {
 public:
  virtual ~Presenter() = default;
  // Blocks until the frame is queued for display (vsync-paced on real displays).
  virtual void present(const uint32_t* bgra, uint32_t width, uint32_t height, uint32_t pitch_pixels) = 0;
  // False once the user closed the window.
  virtual bool alive() const { return true; }
  // Name of the GPU in use, or "" for non-GPU presenters.
  virtual std::string device_name() const { return {}; }
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
