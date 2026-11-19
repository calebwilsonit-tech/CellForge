#include "audio.h"

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

namespace audio {
namespace {

using Clock = std::chrono::steady_clock;

lv2::Kernel& K() { return lv2::Kernel::get(); }

float be_float(const uint8_t* p) {
  const uint32_t v = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
  return std::bit_cast<float>(v);
}

uint64_t now_us() {
  return lv2::system_time_us();
}

// -0.0f as a big-endian word. Written into the first and last sample of every block
// the mixer has consumed; if both are still there when the mixer comes back round,
// the game never refilled the block. (-0.0f == 0.0f, so it still reads as silence.)
constexpr uint32_t kConsumedTag = 0x80000000u;

// Sleeps until a deadline with sub-millisecond accuracy and without spinning.
class Waiter {
 public:
#if defined(_WIN32)
  Waiter() {
    // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION (Windows 10 1803+); plain timer otherwise.
    h_ = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002, TIMER_ALL_ACCESS);
    if (!h_) h_ = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
  }
  ~Waiter() {
    if (h_) CloseHandle(h_);
  }
  void until(Clock::time_point t) {
    const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(t - Clock::now());
    if (d.count() <= 0) return;
    LARGE_INTEGER due;
    due.QuadPart = -int64_t(d.count() / 100);  // relative, 100 ns units
    if (h_ && SetWaitableTimer(h_, &due, 0, nullptr, nullptr, FALSE)) {
      WaitForSingleObject(h_, INFINITE);
      return;
    }
    std::this_thread::sleep_until(t);
  }

 private:
  HANDLE h_ = nullptr;
#else
  void until(Clock::time_point t) { std::this_thread::sleep_until(t); }
#endif
};

}  // namespace

Mixer& Mixer::get() {
  static Mixer m;
  return m;
}

uint64_t Mixer::read_index(uint32_t port) const { return ppu::ld64(K().base(), index_area + 8 * port); }
void Mixer::set_read_index(uint32_t port, uint64_t v) { ppu::st64(K().base(), index_area + 8 * port, v); }

void Mixer::start(std::unique_ptr<Output> out, bool manual) {
  stop();
  out_ = std::move(out);
  started_ = true;
  stop_ = false;
  if (!manual) thread_ = std::thread([this] { run(); });
}

void Mixer::stop() {
  stop_ = true;
  if (thread_.joinable()) thread_.join();
  started_ = false;
}

void Mixer::step() {
  float mix[2 * kBlockFrames] = {};
  std::vector<uint64_t> keys;
  {
    std::lock_guard<std::mutex> lk(m);
    uint8_t* mem = K().base();
    const uint64_t t = now_us();
    for (uint32_t p = 0; p < kMaxPorts; ++p) {
      Port& port = ports[p];
      if (!port.open || !port.running) continue;
      const uint64_t index = read_index(p);
      const uint32_t block = uint32_t(index % port.blocks);
      const uint32_t bytes = port.channels * kBlockFrames * 4;
      const uint32_t base = port.addr + block * bytes;
      if (ppu::ld32(mem, base) == kConsumedTag && ppu::ld32(mem, base + bytes - 4) == kConsumedTag) ++starved;
      const uint8_t* src = mem + base;
      const float lv = port.level;
      for (uint32_t f = 0; f < kBlockFrames; ++f) {
        const uint8_t* fr = src + size_t(f) * port.channels * 4;
        float l = be_float(fr), r = be_float(fr + 4);
        if (port.channels == 8) {
          // FL FR C LFE RL RR SBL SBR -> stereo
          constexpr float k = 0.70710678f;
          const float c = be_float(fr + 8), rl = be_float(fr + 16), rr = be_float(fr + 20);
          const float sbl = be_float(fr + 24), sbr = be_float(fr + 28);
          l += k * (c + rl + sbl);
          r += k * (c + rr + sbr);
        }
        mix[2 * f] += l * lv;
        mix[2 * f + 1] += r * lv;
      }
      // Like cellAudio, clear a block once it has been played: a game that falls
      // behind then produces a gap of silence instead of replaying a stale block.
      std::memset(mem + base, 0, bytes);
      ppu::st32(mem, base, kConsumedTag);
      ppu::st32(mem, base + bytes - 4, kConsumedTag);
      port.block_tag[block] = port.tag;
      port.block_time[block] = t;
      ++port.tag;
      set_read_index(p, (index + 1) % port.blocks);
    }
    keys = notify_keys;
  }
  uint32_t clip = 0;
  for (float& s : mix) {
    if (!(s >= -1.0f && s <= 1.0f)) {  // also catches NaN
      ++clip;
      s = s > 0.0f ? 1.0f : s < 0.0f ? -1.0f : 0.0f;
    }
  }
  if (clip) clipped += clip;
  if (out_) out_->write(mix, kBlockFrames);
  if (static const char* dump_path = std::getenv("PS3RECOMP_DUMP_AUDIO"); dump_path && *dump_path) {
    static FILE* fp = std::fopen(dump_path, "wb");
    if (fp && periods < 15000) {
      int16_t pcm[2 * kBlockFrames];
      for (uint32_t i = 0; i < 2 * kBlockFrames; ++i)
        pcm[i] = int16_t(std::clamp(mix[i], -1.0f, 1.0f) * 32767.0f);
      std::fwrite(pcm, sizeof(int16_t), 2 * kBlockFrames, fp);
      if ((periods & 63) == 0) std::fflush(fp);
    }
  }
  for (uint64_t key : keys) lv2::event_queue_send_by_key(key, 0, 0, 0, 0);
  ++periods;
}

void Mixer::run() {
  using namespace std::chrono;
#if defined(_WIN32)
  // The mixer does very little work per period but must never be starved by the
  // game's busy threads, or every game-side audio consumer stalls with it.
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
  const nanoseconds period(1000000000ll * kBlockFrames / kSampleRate);  // 5.333 ms
  // Device queue level we steer towards (frames). WASAPI drains in ~10 ms quanta, so
  // the level saws by about two blocks; the dead band is wider than that.
  constexpr int64_t kTarget = 8 * kBlockFrames, kBand = 3 * kBlockFrames;
  Waiter waiter;
  auto next = Clock::now();
  auto last_log = next;
  uint64_t log_periods = 0, log_starved = 0, log_clipped = 0, log_late = 0;
  while (!stop_) {
    step();
    if (out_ && out_->paced()) continue;  // a blocking output paces us itself
    nanoseconds p = period;
    const int64_t queued = out_ ? out_->queued_frames() : -1;
    if (queued >= 0) {
      // Follow the device clock by stretching the period by at most 2%; the game
      // only sees a marginally uneven tick, never a burst or a stall.
      if (queued > kTarget + kBand) p += period / 50;
      else if (queued < kTarget - kBand) p -= period / 50;
    }
    next += p;
    const auto now = Clock::now();
    if (now > next + milliseconds(2)) {
      ++late;
      // Badly behind (debugger, system stall): resynchronise rather than burst.
      if (now > next + milliseconds(50)) next = now;
    } else {
      waiter.until(next);
    }
    if (now - last_log >= seconds(10)) {
      const uint64_t pe = periods - log_periods, st = starved - log_starved, cl = clipped - log_clipped,
                     la = late - log_late;
      if (st || cl || la)
        K().logf("audio: last %llds: %llu periods, %llu starved blocks, %llu clipped samples, %llu late periods, "
                 "device queue %lld frames",
                 (long long)duration_cast<seconds>(now - last_log).count(), (unsigned long long)pe,
                 (unsigned long long)st, (unsigned long long)cl, (unsigned long long)la, (long long)queued);
      log_periods = periods, log_starved = starved, log_clipped = clipped, log_late = late;
      last_log = now;
    }
  }
}

}  // namespace audio
