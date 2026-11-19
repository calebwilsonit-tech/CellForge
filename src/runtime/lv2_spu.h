#pragma once
// LV2 SPU thread groups and SPU threads (see lv2_spu.cpp). Exposed so HLE
// libraries (SPURS) can run their own kernels on real SPU thread objects.

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lv2.h"
#include "spu_interp.h"

namespace lv2 {

// sys_spu_segment types
constexpr int32_t kSegCopy = 1, kSegFill = 2, kSegInfo = 4;
// join causes
constexpr uint32_t kJoinGroupExit = 1, kJoinAllThreadsExit = 2, kJoinTerminated = 4;
constexpr uint64_t kSpuThreadEventUserKey = 0xFFFFFFFF53505501ull;
// SPU events
constexpr uint32_t kEventLR = 0x400, kEventS1 = 0x200, kEventS2 = 0x100;
// The SPU thread LS / problem-state window in the effective address space.
constexpr uint32_t kSpuThreadBase = 0xF0000000, kSpuThreadStride = 0x100000;

// Thrown to unwind an SPU host thread when its group is stopped.
struct SpuTerminated {};

struct SpuSegment {
  int32_t type = 0;
  uint32_t ls = 0;
  int32_t size = 0;
  uint32_t addr_or_value = 0;
};

struct SpuGroup;

struct SpuThread : Object, spu::SpuHost {
  uint32_t id = 0;
  std::weak_ptr<SpuGroup> group;
  uint32_t index = 0;
  std::string name;
  spu::SpuCore core{this};
  uint32_t entry = 0;
  std::vector<SpuSegment> segs;
  uint64_t args[4] = {};

  // HLE hooks. hle_main replaces "load image + run" for the whole thread;
  // hle_stop sees every `stop` code first (return true = handled, continue).
  std::function<void(SpuThread&)> hle_main;
  std::function<bool(SpuThread&, uint32_t code, bool& cont)> hle_stop;

  std::mutex m;
  std::condition_variable cv;
  std::deque<uint32_t> in_mbox;   // PPU -> SPU, depth 4
  std::deque<uint32_t> out_mbox;  // SPU -> PPU, depth 1
  uint32_t snr[2] = {};
  bool snr_valid[2] = {};
  bool snr_or[2] = {};
  uint32_t ports[64] = {};        // spup -> event queue id (0 = none)
  std::vector<std::pair<uint32_t, uint32_t>> queues;  // (spuq number, event queue id)

  // MFC
  uint32_t lsa = 0, eah = 0, eal = 0, size = 0, tag = 0, tag_mask = 0;
  uint32_t atomic_stat = 0;
  bool atomic_avail = false;
  uint32_t res_ea = 0;
  bool res_valid = false;
  uint32_t same_getllar = 0;
  uint32_t same_get = 0;
  uint32_t get_cache_lsa[4] = {}, get_cache_ea[4] = {}, get_cache_len[4] = {};
  uint32_t get_cache_next = 0;
  uint8_t res_data[128] = {};
  uint32_t event_mask = 0, event_stat = 0;

  uint32_t exit_status = 0;
  std::thread host;

  void check_stop() {
    if (core.exit_requested()) throw SpuTerminated{};
  }

  // spu::SpuHost
  uint32_t read_channel(spu::SpuCore& s, uint32_t ch) override;
  void write_channel(spu::SpuCore& s, uint32_t ch, uint32_t v) override;
  uint32_t channel_count(spu::SpuCore& s, uint32_t ch) override;
  bool stop(spu::SpuCore& s, uint32_t code) override;

  void mfc_cmd(uint32_t cmd);
  void push_in(uint32_t v);
  uint32_t pop_out();
  bool reservation_lost();
};

struct SpuGroup : Object {
  uint32_t id = 0;
  std::string name;
  uint32_t num = 0;
  int32_t prio = 0;
  uint32_t type = 0;
  std::vector<std::shared_ptr<SpuThread>> threads;

  std::mutex m;
  std::condition_variable cv;
  bool started = false;
  uint32_t running = 0;
  bool done = false;
  uint32_t cause = 0, status = 0;
  uint32_t run_queue = 0;
};

// Kernel-side helpers shared with HLE libraries.
std::shared_ptr<SpuGroup> spu_group_create(const std::string& name, uint32_t num, int32_t prio);
std::shared_ptr<SpuThread> spu_thread_create(const std::shared_ptr<SpuGroup>& g, uint32_t index, const std::string& name);
uint32_t spu_group_start(uint32_t group_id);
void spu_group_finish(SpuGroup& g, uint32_t cause, uint32_t status);
uint32_t spu_group_join(uint32_t group_id, uint32_t* cause, uint32_t* status);
// Lock serialising 128-byte lock-line operations (GETLLAR / PUTLLC / PUTLLUC).
std::mutex& spu_line_mutex();

}  // namespace lv2
