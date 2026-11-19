// Smoke test for the Vulkan presenter: opens a window, reports the GPU, and shows an
// animated pattern for a few seconds (or until the window is closed).

#include <chrono>
#include <cstdio>
#include <vector>

#include "presenter.h"

int main() {
  auto log = [](std::string_view s) { std::printf("%.*s\n", int(s.size()), s.data()); };
  auto p = gfx::create_vulkan_presenter("ps3recomp - Vulkan test", log);
  if (!p) {
    std::printf("FAILED: could not create the Vulkan presenter\n");
    return 1;
  }
  std::printf("GPU: %s\n", p->device_name().c_str());

  const uint32_t w = 1280, h = 720;
  std::vector<uint32_t> px(size_t(w) * h);
  const auto start = std::chrono::steady_clock::now();
  uint32_t frames = 0;
  while (p->alive()) {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (t > 3.0) break;
    const uint32_t shift = uint32_t(t * 200);
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x) {
        const uint32_t r = (x + shift) & 0xFF, g = (y + shift / 2) & 0xFF, b = ((x ^ y) >> 1) & 0xFF;
        px[size_t(y) * w + x] = 0xFF000000u | (r << 16) | (g << 8) | b;  // host BGRA
      }
    p->present(px.data(), w, h, w);
    ++frames;
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("presented %u frames in %.2f s (%.1f fps)\n", frames, secs, frames / secs);
  return 0;
}
