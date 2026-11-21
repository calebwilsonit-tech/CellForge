#pragma once
// LV2: a host implementation of the PS3 kernel services that recompiled code
// reaches through `sc` (syscalls) and through the HLE system libraries.
//
// Guest threads are real host threads, each with its own ppu::Context.
// Kernel objects (mutexes, condition variables, semaphores, event queues) are
// implemented with std::mutex / std::condition_variable.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>

#include "guest_memory.h"
#include "ppu_runtime.h"

namespace lv2 {

// ---- error codes ------------------------------------------------------------
constexpr uint32_t CELL_OK = 0;
constexpr uint32_t CELL_EAGAIN = 0x80010001;
constexpr uint32_t CELL_EINVAL = 0x80010002;
constexpr uint32_t CELL_ENOSYS = 0x80010003;
constexpr uint32_t CELL_ENOMEM = 0x80010004;
constexpr uint32_t CELL_ESRCH = 0x80010005;
constexpr uint32_t CELL_ENOENT = 0x80010006;
constexpr uint32_t CELL_ENOEXEC = 0x80010007;
constexpr uint32_t CELL_EDEADLK = 0x80010008;
constexpr uint32_t CELL_EPERM = 0x80010009;
constexpr uint32_t CELL_EBUSY = 0x8001000A;
constexpr uint32_t CELL_ETIMEDOUT = 0x8001000B;
constexpr uint32_t CELL_EFAULT = 0x8001000D;
constexpr uint32_t CELL_ECANCELED = 0x80010013;
constexpr uint32_t CELL_EISCONN = 0x80010016;
constexpr uint32_t CELL_ENOTCONN = 0x80010017;

constexpr uint64_t kTimebaseFrequency = 79800000;

// Shared monotonic system time in microseconds (matches sys_time_get_system_time,
// cellAudioGetPortTimestamp, and cellGcmGetLastFlipTime).
inline uint64_t system_time_us() {
  using namespace std::chrono;
  static const auto start = steady_clock::now();
  return uint64_t(duration_cast<microseconds>(steady_clock::now() - start).count()) + 1;
}

// Thrown to unwind a guest thread / the process back to the kernel.
struct ThreadExit { uint64_t value; };
struct ProcessExit { int status; };

// ---- guest memory regions -----------------------------------------------------
class GuestAllocator {
 public:
  void init(ppu::GuestMemory* mem, uint32_t base, uint32_t size);
  uint32_t alloc(uint32_t size, uint32_t align);  // 0 on failure
  bool free(uint32_t addr);
  uint32_t total() const { return size_; }
  uint32_t used() const;

 private:
  mutable std::mutex m_;
  ppu::GuestMemory* mem_ = nullptr;
  uint32_t base_ = 0, size_ = 0;
  std::map<uint32_t, uint32_t> free_;   // addr -> size
  std::map<uint32_t, uint32_t> used_;   // addr -> size
};

// Address map (all inside the 4 GiB guest space):
//   ELF segments        wherever the executable says (typically 0x00010000+)
//   0x2F000000 - HLE    function descriptors, strings and other loader data
//   0x30000000 - heap   sys_memory_allocate / _sys_malloc
//   0xD0000000 - stacks thread stacks and TLS blocks
constexpr uint32_t kHleBase = 0x2F000000, kHleSize = 0x01000000;
constexpr uint32_t kHeapBase = 0x30000000, kHeapSize = 0x0D500000;  // ~213 MB, as on retail
constexpr uint32_t kStackBase = 0xD0000000, kStackSize = 0x10000000;

// ---- threads -------------------------------------------------------------------
struct Thread {
  uint64_t id = 0;
  std::string name;
  ppu::Context ctx;
  uint32_t entry = 0;  // code address
  uint32_t stack_addr = 0, stack_size = 0, tls_block = 0;
  int32_t prio = 1001;
  bool joinable = true;

  std::mutex m;
  std::condition_variable cv;
  bool started = false, finished = false, joined = false;
  uint64_t exit_value = 0;
  std::thread host;
  // Diagnostics: syscall in progress (n | 0x80000000, 0 = none) and its r3/lr.
  std::atomic<uint32_t> cur_sys{0};
  std::atomic<uint32_t> cur_sys_r3{0}, cur_sys_lr{0};
};

struct Object {
  virtual ~Object() = default;
};

class Kernel {
 public:
  static Kernel& get();

  void init(ppu::GuestMemory* mem);
  ppu::GuestMemory& memory() { return *mem_; }
  uint8_t* base() { return mem_->base(); }

  // memory
  GuestAllocator heap, stacks, hle;

  // process
  uint32_t sdk_version = 0x00360001;
  std::function<void(std::string_view)> tty;  // sys_tty_write / _sys_printf output
  std::function<void(std::string_view)> log;  // runtime diagnostics
  bool strict = false;                        // throw on unimplemented syscalls
  void logf(const char* fmt, ...);
  void dump_context(const ppu::Context& c);  // registers + guest call trace, for fatal errors
  std::string guest_trace(uint32_t sp);       // saved LRs along the back chain
  // PS3RECOMP_THREAD_DUMP=<sec>: log every guest thread's state every <sec> seconds.
  void dump_threads();
  void start_thread_dumper();

  // TLS
  void set_tls_image(uint32_t addr, uint32_t filesz, uint32_t memsz);
  uint32_t alloc_tls();  // returns the r13 value for a new thread

  // threads
  std::shared_ptr<Thread> create_thread(uint32_t entry_opd, uint64_t arg, int32_t prio, uint32_t stack_size,
                                        bool joinable, std::string name);
  uint32_t start_thread(uint64_t id);
  uint32_t join_thread(uint64_t id, uint64_t* exit_value);
  uint32_t detach_thread(uint64_t id);
  std::shared_ptr<Thread> find_thread(uint64_t id);
  // Runs `t` on the calling host thread (used for the primary thread).
  void run_here(const std::shared_ptr<Thread>& t);
  // The guest thread running on this host thread (attaches one if needed).
  Thread& current();
  void detach_all_threads();
  // Runs a guest function (given its OPD) on the calling host thread, for example
  // RSX flip and vblank handlers. Each host thread gets its own small guest stack.
  uint64_t call_guest(uint32_t opd, std::initializer_list<uint64_t> args);

  // kernel objects
  uint32_t add_object(std::shared_ptr<Object> obj);
  template <class T> std::shared_ptr<T> get_object(uint32_t id) {
    std::lock_guard<std::mutex> lk(objects_m_);
    auto it = objects_.find(id);
    return it == objects_.end() ? nullptr : std::dynamic_pointer_cast<T>(it->second);
  }
  bool remove_object(uint32_t id);
  template <class F> void for_each_object(F f) {
    std::lock_guard<std::mutex> lk(objects_m_);
    for (auto& [id, o] : objects_) f(id, o);
  }

  // syscalls (installed as ppu::syscall_handler by init)
  void syscall(ppu::Context& c);

  // guest memory helpers
  uint16_t r16(uint32_t a) { return ppu::ld16(base(), a); }
  void w16(uint32_t a, uint16_t v) { if (a) ppu::st16(base(), a, v); }
  uint32_t r32(uint32_t a) { return ppu::ld32(base(), a); }
  uint64_t r64(uint32_t a) { return ppu::ld64(base(), a); }
  void w32(uint32_t a, uint32_t v) { if (a) ppu::st32(base(), a, v); }
  void w64(uint32_t a, uint64_t v) { if (a) ppu::st64(base(), a, v); }
  std::string read_str(uint32_t a, size_t max = 4096);
  uint32_t alloc_str(std::string_view s);  // copies into the HLE region

 private:
  Kernel() = default;
  void run_thread(Thread* t);

  ppu::GuestMemory* mem_ = nullptr;
  uint32_t tls_addr_ = 0, tls_filesz_ = 0, tls_memsz_ = 0;

  std::mutex threads_m_;
  std::unordered_map<uint64_t, std::shared_ptr<Thread>> threads_;
  uint64_t next_thread_id_ = 0x01000000;

  std::mutex objects_m_;
  std::unordered_map<uint32_t, std::shared_ptr<Object>> objects_;
  uint32_t next_object_id_ = 0x40000001;
};

// ---- synchronisation objects (also used by the HLE lwmutex/lwcond) ---------------
struct Mutex : Object {
  std::mutex m;
  std::condition_variable cv;
  uint64_t owner = 0;
  uint32_t count = 0;
  bool recursive = false;
  uint32_t cond_refs = 0;
};

uint32_t mutex_create(bool recursive, uint32_t* out_id);
uint32_t mutex_lock(uint32_t id, uint64_t timeout_us);
uint32_t mutex_trylock(uint32_t id);
uint32_t mutex_unlock(uint32_t id);
uint32_t mutex_destroy(uint32_t id);
uint32_t cond_create(uint32_t mutex_id, uint32_t* out_id);
uint32_t cond_wait(uint32_t id, uint64_t timeout_us);
uint32_t cond_signal(uint32_t id);
uint32_t cond_signal_all(uint32_t id);
uint32_t cond_signal_to(uint32_t id, uint64_t thread_id);
uint32_t cond_destroy(uint32_t id);

// Event queues used by HLE libraries (e.g. cellAudio's block notifications).
uint32_t event_queue_create(int32_t size, uint64_t key, uint32_t* out_id);
// Sends an event to the queue with `key`. Returns false if no such queue / queue full.
bool event_queue_send_by_key(uint64_t key, uint64_t source, uint64_t d1, uint64_t d2, uint64_t d3);
// By queue id. send: CELL_OK / CELL_ESRCH / CELL_EBUSY (full).
uint32_t event_queue_send(uint32_t id, uint64_t source, uint64_t d1, uint64_t d2, uint64_t d3);
// out = {source, d1, d2, d3}. wait=false: CELL_EBUSY when empty.
uint32_t event_queue_receive(uint32_t id, uint64_t out[4], bool wait);

// SPU thread groups (lv2_spu.cpp): syscalls 169-194, 251/252. Returns false if
// `n` is not an SPU syscall.
bool spu_syscall(uint32_t n, ppu::Context& c, uint64_t& rc);
void register_spu_modules();  // sys_spu_image_import / close

// sys_mmapper: address areas and memory objects (syscalls 329-337 and the
// sysPrxForUser wrappers).
uint32_t mmapper_allocate_address(uint32_t size, uint32_t align, uint32_t* addr);
uint32_t mmapper_allocate_memory(uint32_t size, uint32_t* id);
uint32_t mmapper_free_memory(uint32_t id);
uint32_t mmapper_map_memory(uint32_t addr, uint32_t id);
uint32_t mmapper_search_and_map(uint32_t start, uint32_t id, uint32_t* out);
uint32_t mmapper_unmap_memory(uint32_t addr, uint32_t* id);

// Name of a syscall number, for logs ("" if unknown).
const char* syscall_name(uint32_t n);

// Logs the next `count` syscalls from any PPU thread (used by mods for troubleshooting).
void trace_next_syscalls(uint32_t count);
}  // namespace lv2
