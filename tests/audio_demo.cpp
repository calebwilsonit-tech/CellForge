// Smoke test for the host audio output: plays a quiet 440 Hz tone for about a second.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

#include "audio.h"

int main() {
  auto out = audio::create_wasapi_output();
  if (!out) {
    std::printf("FAILED: no audio output backend\n");
    return 1;
  }
  const uint32_t frames = audio::kBlockFrames;
  std::vector<float> buf(2 * frames);
  const auto start = std::chrono::steady_clock::now();
  double phase = 0;
  for (int block = 0; block < int(audio::kSampleRate / frames); ++block) {
    for (uint32_t f = 0; f < frames; ++f) {
      const float s = 0.05f * float(std::sin(phase));
      phase += 2 * 3.14159265358979 * 440.0 / audio::kSampleRate;
      buf[2 * f] = buf[2 * f + 1] = s;
    }
    out->write(buf.data(), frames);
    // Outputs no longer block; pace like the mixer does (one block per 5.33 ms).
    std::this_thread::sleep_until(start + std::chrono::microseconds(1000000ll * frames * (block + 1) / audio::kSampleRate));
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("device active: %s, wrote 1.0 s of audio in %.2f s\n", out->active() ? "yes" : "no", secs);
  return out->active() ? 0 : 1;
}
