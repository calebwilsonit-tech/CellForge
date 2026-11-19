#pragma once
// PS3 audio as seen through cellAudio: up to 4 ports, each a ring of blocks of
// 256 sample frames (float32, big-endian, 2 or 8 interleaved channels) in guest
// memory. Every 256 frames at 48 kHz (5.33 ms) the mixer reads the current block of
// every running port, clears it, advances its read index, notifies the registered
// event queues and sends a stereo mix to the host audio device. The period is kept
// on a precise host timer (not the device's 10 ms quantum) so the game sees the same
// even cadence real hardware gives it; the device queue is only used to correct drift.

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace audio {

constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kBlockFrames = 256;
constexpr uint32_t kMaxPorts = 4;

// Where mixed stereo frames go.
class Output {
 public:
  virtual ~Output() = default;
  // Blocks until the frames are queued (this paces the mixer on real devices).
  virtual void write(const float* stereo, uint32_t frames) = 0;
  // True if write() blocks at the device rate; otherwise the mixer paces itself.
  virtual bool paced() const { return false; }
  // True once a real output device is open and playing.
  virtual bool active() const { return false; }
  // Frames queued on the device but not yet played, or -1 if unknown. The mixer
  // uses this to follow the device clock without ever blocking on it.
  virtual int64_t queued_frames() const { return -1; }
};

std::unique_ptr<Output> create_wasapi_output();  // Windows default device; nullptr if unavailable

// Keeps everything written (tests).
class CaptureOutput : public Output {
 public:
  void write(const float* s, uint32_t frames) override {
    std::lock_guard<std::mutex> lk(m);
    samples.insert(samples.end(), s, s + 2 * frames);
  }
  std::mutex m;
  std::vector<float> samples;
};

struct Port {
  bool open = false, running = false;
  uint32_t channels = 2, blocks = 8;
  uint32_t addr = 0, size = 0;
  float level = 1.0f;
  uint64_t tag = 0;  // number of blocks mixed so far
  std::array<uint64_t, 32> block_tag{}, block_time{};
};

class Mixer {
 public:
  static Mixer& get();

  // Starts the mixer thread unless `manual` (tests call step() instead).
  void start(std::unique_ptr<Output> out, bool manual = false);
  void stop();
  bool started() const { return started_; }

  // Mixes one block period. Called by the thread or by tests.
  void step();

  std::mutex m;  // guards ports, keys and the index area
  Port ports[kMaxPorts];
  uint32_t index_area = 0;  // guest address: u64 read index per port
  std::vector<uint64_t> notify_keys;
  std::atomic<uint64_t> periods{0};
  // Diagnostics: blocks the game had not refilled when the mixer reached them,
  // output samples that hit the clip limit, and mixer periods that ran late (>2 ms).
  std::atomic<uint64_t> starved{0}, clipped{0}, late{0};

  uint64_t read_index(uint32_t port) const;
  void set_read_index(uint32_t port, uint64_t v);

 private:
  Mixer() = default;
  ~Mixer() { stop(); }
  void run();
  std::unique_ptr<Output> out_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  bool started_ = false;
};

}  // namespace audio
