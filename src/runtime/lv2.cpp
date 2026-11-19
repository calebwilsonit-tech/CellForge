#define _CRT_SECURE_NO_WARNINGS  // getenv for the syscall trace option
#include "lv2.h"
#include "profiler.h"
#include "lv2_spu.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <set>
#include <unordered_set>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace lv2 {

using Clock = std::chrono::steady_clock;

namespace {

thread_local Thread* t_current = nullptr;

// Waits on cv until pred() or the timeout (microseconds, 0 = forever) expires.
template <class Lock, class Pred>
bool wait_for(std::condition_variable& cv, Lock& lk, uint64_t timeout_us, Pred pred) {
  if (timeout_us == 0) { cv.wait(lk, pred); return true; }
  return cv.wait_for(lk, std::chrono::microseconds(timeout_us), pred);
}

uint32_t align_up(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// Games poll with very short sleeps (sys_timer_usleep(30) is common). A plain host
// sleep overshoots by 1-2 ms, which multiplies every polling loop's latency; but
// spinning through every sleep (as this used to) kept several host cores pegged
// for threads that were only waiting. So:
//  - a thread's first few short sleeps in a row spin briefly (a quick wait stays quick);
//  - a thread that keeps polling, and every longer sleep, blocks on a high-resolution
//    waitable timer (about 0.55 ms late on Windows 10+), only spinning for whatever
//    is left if the timer fires early.
#if defined(_WIN32)
void timer_wait(int64_t us) {
  thread_local HANDLE timer = [] {
    HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /*HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
    return h ? h : CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
  }();
  LARGE_INTEGER due;
  due.QuadPart = -std::max<int64_t>(us, 1) * 10;
  if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, INFINITE);
  else std::this_thread::sleep_for(std::chrono::microseconds(us));
}
#else
void timer_wait(int64_t us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }
#endif

void precise_sleep(uint64_t us) {
  using namespace std::chrono;
  if (us <= 2) {
    std::this_thread::yield();
    return;
  }
  const auto now = steady_clock::now();
  if (us <= 300) {
    // Count consecutive short sleeps (a gap of >2 ms means the thread did real work).
    thread_local steady_clock::time_point last_t{};
    thread_local uint32_t streak = 0;
    streak = (now - last_t < milliseconds(2)) ? streak + 1 : 0;
    last_t = now;
    if (streak < 24) {
      const auto deadline = now + microseconds(std::min<uint64_t>(us, 15));
      while (steady_clock::now() < deadline) std::this_thread::yield();
    } else {
      timer_wait(1);  // still polling: give the core back (~0.55 ms)
      last_t = steady_clock::now();
    }
    return;
  }
  constexpr int64_t kTimerLate = 500;  // typical high-resolution timer overshoot
  const auto deadline = now + microseconds(us);
  timer_wait(std::max<int64_t>(int64_t(us) - kTimerLate, 1));
  while (steady_clock::now() < deadline) std::this_thread::yield();
}

}  // namespace

// ============================================================================
// GuestAllocator
// ============================================================================
void GuestAllocator::init(ppu::GuestMemory* mem, uint32_t base, uint32_t size) {
  std::lock_guard<std::mutex> lk(m_);
  mem_ = mem;
  base_ = base;
  size_ = size;
  free_.clear();
  used_.clear();
  free_[base] = size;
}

uint32_t GuestAllocator::alloc(uint32_t size, uint32_t align) {
  if (!size) size = 1;
  if (align < 16) align = 16;
  size = align_up(size, 16);
  std::lock_guard<std::mutex> lk(m_);
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    const uint32_t start = it->first, len = it->second;
    const uint32_t a = align_up(start, align);
    if (a < start || uint64_t(a) + size > uint64_t(start) + len) continue;
    free_.erase(it);
    if (a > start) free_[start] = a - start;
    const uint64_t end = uint64_t(a) + size, blk_end = uint64_t(start) + len;
    if (end < blk_end) free_[uint32_t(end)] = uint32_t(blk_end - end);
    used_[a] = size;
    if (mem_) {
      mem_->map(a, size);
      std::memset(mem_->base() + a, 0, size);
    }
    return a;
  }
  return 0;
}

bool GuestAllocator::free(uint32_t addr) {
  std::lock_guard<std::mutex> lk(m_);
  auto it = used_.find(addr);
  if (it == used_.end()) return false;
  uint32_t start = addr, len = it->second;
  used_.erase(it);
  auto next = free_.lower_bound(start);
  if (next != free_.end() && start + len == next->first) { len += next->second; next = free_.erase(next); }
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == start) { start = prev->first; len += prev->second; free_.erase(prev); }
  }
  free_[start] = len;
  return true;
}

uint32_t GuestAllocator::used() const {
  std::lock_guard<std::mutex> lk(m_);
  uint32_t n = 0;
  for (const auto& [a, s] : used_) n += s;
  return n;
}

// ============================================================================
// Kernel basics
// ============================================================================
Kernel& Kernel::get() {
  static Kernel k;
  return k;
}

void Kernel::init(ppu::GuestMemory* mem) {
  mem_ = mem;
  heap.init(mem, kHeapBase, kHeapSize);
  stacks.init(mem, kStackBase, kStackSize);
  hle.init(mem, kHleBase, kHleSize);
  ppu::syscall_handler = [](ppu::Context& c) { Kernel::get().syscall(c); };
}

void Kernel::logf(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  if (log) log(buf);
}

std::string Kernel::read_str(uint32_t a, size_t max) {
  std::string s;
  if (!a) return s;
  for (size_t i = 0; i < max; ++i) {
    const char ch = char(base()[a + i]);
    if (!ch) break;
    s += ch;
  }
  return s;
}

uint32_t Kernel::alloc_str(std::string_view s) {
  const uint32_t a = hle.alloc(uint32_t(s.size() + 1), 16);
  if (a) std::memcpy(base() + a, s.data(), s.size());
  return a;
}

void Kernel::set_tls_image(uint32_t addr, uint32_t filesz, uint32_t memsz) {
  tls_addr_ = addr;
  tls_filesz_ = filesz;
  tls_memsz_ = memsz;
}

// PPC64 TLS: r13 points 0x7000 past the start of the block; the PS3 keeps 0x30
// bytes of thread data in front of the TLS image.
uint32_t Kernel::alloc_tls() {
  const uint32_t block = stacks.alloc(0x30 + tls_memsz_ + 0x10, 16);
  if (!block) return 0;
  if (tls_filesz_) std::memcpy(base() + block + 0x30, base() + tls_addr_, tls_filesz_);
  return block + 0x7030;
}

// ============================================================================
// Threads
// ============================================================================
std::shared_ptr<Thread> Kernel::create_thread(uint32_t entry_opd, uint64_t arg, int32_t prio, uint32_t stack_size,
                                              bool joinable, std::string name) {
  auto t = std::make_shared<Thread>();
  t->name = std::move(name);
  t->prio = prio;
  t->joinable = joinable;
  stack_size = align_up(stack_size < 0x4000 ? 0x4000 : stack_size, 0x1000);
  t->stack_size = stack_size;
  t->stack_addr = stacks.alloc(stack_size, 0x1000);
  if (!t->stack_addr) return nullptr;
  const uint32_t r13 = alloc_tls();

  ppu::Context& c = t->ctx;
  c.base = base();
  t->entry = r32(entry_opd);
  c.r[2] = r32(entry_opd + 4);
  c.r[1] = t->stack_addr + stack_size - 0x70;
  c.r[3] = arg;
  c.r[13] = r13;

  std::lock_guard<std::mutex> lk(threads_m_);
  t->id = next_thread_id_++;
  threads_[t->id] = t;
  return t;
}

void apply_host_priority(int32_t prio) {
#if defined(_WIN32)
  // PS3 PPU priorities: 0 (highest) .. 3071 (lowest). Elevate real-time audio/IO
  // threads (e.g. RWAC Dac) so they aren't starved by rendering/SPU worker threads.
  const int p = prio < 500 ? THREAD_PRIORITY_HIGHEST
                : prio < 1000 ? THREAD_PRIORITY_ABOVE_NORMAL
                : prio > 2000 ? THREAD_PRIORITY_BELOW_NORMAL
                              : THREAD_PRIORITY_NORMAL;
  SetThreadPriority(GetCurrentThread(), p);
#else
  (void)prio;
#endif
}

void Kernel::run_thread(Thread* t) {
  t_current = t;
  apply_host_priority(t->prio);
  prof::register_thread("ppu:" + t->name);
  uint64_t value = 0;
  try {
    ppu::call_indirect(t->ctx, t->entry);
    value = t->ctx.r[3];
  } catch (const ThreadExit& e) {
    value = e.value;
  } catch (const ProcessExit& e) {
    logf("process exit(%d) from thread '%s'", e.status, t->name.c_str());
    std::fflush(nullptr);
    std::_Exit(e.status);
  } catch (const std::exception& e) {
    logf("fatal error in thread '%s': %s", t->name.c_str(), e.what());
    dump_context(t->ctx);
    std::fflush(nullptr);
    std::_Exit(1);
  }
  std::lock_guard<std::mutex> lk(t->m);
  t->exit_value = value;
  t->finished = true;
  t->cv.notify_all();
}

uint32_t Kernel::start_thread(uint64_t id) {
  auto t = find_thread(id);
  if (!t) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(t->m);
  if (t->started) return CELL_EBUSY;
  t->started = true;
  t->host = std::thread([this, raw = t.get()] { run_thread(raw); });
  if (!t->joinable) t->host.detach();
  return CELL_OK;
}

void Kernel::run_here(const std::shared_ptr<Thread>& t) {
  {
    std::lock_guard<std::mutex> lk(t->m);
    t->started = true;
  }
  t_current = t.get();
  prof::register_thread("ppu:" + t->name);
  // Exceptions (ProcessExit) propagate to the caller for the primary thread.
  ppu::call_indirect(t->ctx, t->entry);
  std::lock_guard<std::mutex> lk(t->m);
  t->exit_value = t->ctx.r[3];
  t->finished = true;
  t->cv.notify_all();
}

uint32_t Kernel::join_thread(uint64_t id, uint64_t* exit_value) {
  auto t = find_thread(id);
  if (!t) return CELL_ESRCH;
  if (t.get() == t_current) return CELL_EDEADLK;
  {
    std::unique_lock<std::mutex> lk(t->m);
    if (!t->joinable || t->joined) return CELL_EINVAL;
    t->joined = true;
    t->cv.wait(lk, [&] { return t->finished; });
    if (exit_value) *exit_value = t->exit_value;
  }
  if (t->host.joinable()) t->host.join();
  std::lock_guard<std::mutex> lk(threads_m_);
  threads_.erase(id);
  return CELL_OK;
}

uint32_t Kernel::detach_thread(uint64_t id) {
  auto t = find_thread(id);
  if (!t) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(t->m);
  if (!t->joinable) return CELL_EINVAL;
  t->joinable = false;
  if (t->host.joinable()) t->host.detach();
  return CELL_OK;
}

std::shared_ptr<Thread> Kernel::find_thread(uint64_t id) {
  std::lock_guard<std::mutex> lk(threads_m_);
  auto it = threads_.find(id);
  return it == threads_.end() ? nullptr : it->second;
}

Thread& Kernel::current() {
  if (t_current) return *t_current;
  // A host thread that isn't a guest thread (tests, host tools): give it an identity.
  auto t = std::make_shared<Thread>();
  t->name = "host";
  t->ctx.base = mem_ ? base() : nullptr;
  {
    std::lock_guard<std::mutex> lk(threads_m_);
    t->id = next_thread_id_++;
    threads_[t->id] = t;
  }
  t->started = true;
  t_current = t.get();
  return *t;
}

uint64_t Kernel::call_guest(uint32_t opd, std::initializer_list<uint64_t> args) {
  thread_local std::shared_ptr<Thread> host_cb;
  if (!host_cb) {
    auto t = std::make_shared<Thread>();
    t->name = "callback";
    t->stack_size = 0x10000;
    t->stack_addr = stacks.alloc(t->stack_size, 0x1000);
    t->tls_block = alloc_tls();
    t->started = true;
    {
      std::lock_guard<std::mutex> lk(threads_m_);
      t->id = next_thread_id_++;
      threads_[t->id] = t;
    }
    host_cb = t;
  }
  Thread* const saved = t_current;
  t_current = host_cb.get();
  ppu::Context& ctx = host_cb->ctx;
  ctx = ppu::Context{};
  ctx.base = base();
  ctx.r[1] = host_cb->stack_addr + host_cb->stack_size - 0x70;
  ctx.r[13] = host_cb->tls_block;
  ctx.r[2] = r32(opd + 4);
  int reg = 3;
  for (uint64_t a : args) {
    if (reg > 10) break;
    ctx.r[reg++] = a;
  }
  try {
    ppu::call_indirect(ctx, r32(opd));
  } catch (...) {
    t_current = saved;
    throw;
  }
  t_current = saved;
  return ctx.r[3];
}

void Kernel::detach_all_threads() {
  std::lock_guard<std::mutex> lk(threads_m_);
  for (auto& [id, t] : threads_)
    if (t->host.joinable()) t->host.detach();
}

// ============================================================================
// Objects
// ============================================================================
uint32_t Kernel::add_object(std::shared_ptr<Object> obj) {
  std::lock_guard<std::mutex> lk(objects_m_);
  const uint32_t id = next_object_id_++;
  objects_[id] = std::move(obj);
  return id;
}

bool Kernel::remove_object(uint32_t id) {
  std::lock_guard<std::mutex> lk(objects_m_);
  return objects_.erase(id) != 0;
}

// ---- mutex / cond --------------------------------------------------------------
namespace {

struct Cond : Object {
  std::shared_ptr<Mutex> mtx;
  uint32_t mutex_id = 0;
  std::condition_variable cv;
  uint32_t waiters = 0, signals = 0;
  std::unordered_set<uint64_t> targeted;
};

struct Semaphore : Object {
  std::mutex m;
  std::condition_variable cv;
  int32_t value = 0, max = 0;
};

struct Event {
  uint64_t source, d1, d2, d3;
};

struct EventQueue : Object {
  std::mutex m;
  std::condition_variable cv;
  std::deque<Event> events;
  int32_t size = 127;
  uint64_t key = 0;
  uint32_t waiters = 0;
  bool destroyed = false;
};

struct EventPort : Object {
  std::mutex m;
  std::weak_ptr<EventQueue> queue;
  uint32_t queue_id = 0;
  uint64_t name = 0;
};

Kernel& K() { return Kernel::get(); }
uint64_t me() { return K().current().id; }

}  // namespace

uint32_t mutex_create(bool recursive, uint32_t* out_id) {
  auto m = std::make_shared<Mutex>();
  m->recursive = recursive;
  *out_id = K().add_object(m);
  return CELL_OK;
}

uint32_t mutex_lock(uint32_t id, uint64_t timeout_us) {
  auto m = K().get_object<Mutex>(id);
  if (!m) return CELL_ESRCH;
  const uint64_t t = me();
  std::unique_lock<std::mutex> lk(m->m);
  if (m->owner == t) {
    if (!m->recursive) return CELL_EDEADLK;
    ++m->count;
    return CELL_OK;
  }
  if (!wait_for(m->cv, lk, timeout_us, [&] { return m->owner == 0; })) return CELL_ETIMEDOUT;
  m->owner = t;
  m->count = 1;
  return CELL_OK;
}

uint32_t mutex_trylock(uint32_t id) {
  auto m = K().get_object<Mutex>(id);
  if (!m) return CELL_ESRCH;
  const uint64_t t = me();
  std::lock_guard<std::mutex> lk(m->m);
  if (m->owner == t) {
    if (!m->recursive) return CELL_EDEADLK;
    ++m->count;
    return CELL_OK;
  }
  if (m->owner) return CELL_EBUSY;
  m->owner = t;
  m->count = 1;
  return CELL_OK;
}

uint32_t mutex_unlock(uint32_t id) {
  auto m = K().get_object<Mutex>(id);
  if (!m) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(m->m);
  if (m->owner != me()) return CELL_EPERM;
  if (--m->count == 0) {
    m->owner = 0;
    m->cv.notify_all();
  }
  return CELL_OK;
}

uint32_t mutex_destroy(uint32_t id) {
  auto m = K().get_object<Mutex>(id);
  if (!m) return CELL_ESRCH;
  {
    std::lock_guard<std::mutex> lk(m->m);
    if (m->owner) return CELL_EBUSY;
    if (m->cond_refs) return CELL_EPERM;
  }
  K().remove_object(id);
  return CELL_OK;
}

uint32_t cond_create(uint32_t mutex_id, uint32_t* out_id) {
  auto m = K().get_object<Mutex>(mutex_id);
  if (!m) return CELL_ESRCH;
  auto c = std::make_shared<Cond>();
  c->mtx = m;
  c->mutex_id = mutex_id;
  {
    std::lock_guard<std::mutex> lk(m->m);
    ++m->cond_refs;
  }
  *out_id = K().add_object(c);
  return CELL_OK;
}

uint32_t cond_wait(uint32_t id, uint64_t timeout_us) {
  auto c = K().get_object<Cond>(id);
  if (!c) return CELL_ESRCH;
  Mutex& m = *c->mtx;
  const uint64_t t = me();
  std::unique_lock<std::mutex> lk(m.m);
  if (m.owner != t) return CELL_EPERM;
  const uint32_t saved = m.count;
  m.owner = 0;
  m.count = 0;
  m.cv.notify_all();

  ++c->waiters;
  const bool signaled = wait_for(c->cv, lk, timeout_us, [&] { return c->signals > 0 || c->targeted.count(t); });
  if (signaled) {
    if (c->targeted.erase(t) == 0) --c->signals;
  }
  --c->waiters;

  m.cv.wait(lk, [&] { return m.owner == 0; });
  m.owner = t;
  m.count = saved;
  return signaled ? CELL_OK : CELL_ETIMEDOUT;
}

uint32_t cond_signal(uint32_t id) {
  auto c = K().get_object<Cond>(id);
  if (!c) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(c->mtx->m);
  if (c->signals + c->targeted.size() < c->waiters) {
    ++c->signals;
    c->cv.notify_all();
  }
  return CELL_OK;
}

uint32_t cond_signal_all(uint32_t id) {
  auto c = K().get_object<Cond>(id);
  if (!c) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(c->mtx->m);
  c->signals = c->waiters;
  c->cv.notify_all();
  return CELL_OK;
}

uint32_t cond_signal_to(uint32_t id, uint64_t thread_id) {
  auto c = K().get_object<Cond>(id);
  if (!c) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(c->mtx->m);
  c->targeted.insert(thread_id);
  c->cv.notify_all();
  return CELL_OK;
}

uint32_t cond_destroy(uint32_t id) {
  auto c = K().get_object<Cond>(id);
  if (!c) return CELL_ESRCH;
  {
    std::lock_guard<std::mutex> lk(c->mtx->m);
    if (c->waiters) return CELL_EBUSY;
    --c->mtx->cond_refs;
  }
  K().remove_object(id);
  return CELL_OK;
}

// ============================================================================
// Syscalls
// ============================================================================
namespace {

#define ARG(n) c.r[3 + (n)]
#define A32(n) uint32_t(c.r[3 + (n)])

uint64_t sys_semaphore_create(ppu::Context& c) {
  const int32_t initial = int32_t(ARG(2)), max = int32_t(ARG(3));
  if (max <= 0 || initial < 0 || initial > max) return CELL_EINVAL;
  auto s = std::make_shared<Semaphore>();
  s->value = initial;
  s->max = max;
  K().w32(A32(0), K().add_object(s));
  return CELL_OK;
}
uint64_t sys_semaphore_wait(ppu::Context& c, bool try_only) {
  auto s = K().get_object<Semaphore>(A32(0));
  if (!s) return CELL_ESRCH;
  std::unique_lock<std::mutex> lk(s->m);
  if (try_only) {
    if (s->value <= 0) return CELL_EBUSY;
  } else if (!wait_for(s->cv, lk, ARG(1), [&] { return s->value > 0; })) {
    return CELL_ETIMEDOUT;
  }
  --s->value;
  return CELL_OK;
}
uint64_t sys_semaphore_post(ppu::Context& c) {
  auto s = K().get_object<Semaphore>(A32(0));
  if (!s) return CELL_ESRCH;
  const int32_t n = int32_t(ARG(1));
  if (n < 0) return CELL_EINVAL;
  std::lock_guard<std::mutex> lk(s->m);
  if (int64_t(s->value) + n > s->max) return CELL_EBUSY;
  s->value += n;
  s->cv.notify_all();
  return CELL_OK;
}
uint64_t sys_semaphore_get_value(ppu::Context& c) {
  auto s = K().get_object<Semaphore>(A32(0));
  if (!s) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(s->m);
  K().w32(A32(1), uint32_t(s->value));
  return CELL_OK;
}

uint64_t sys_mutex_create(ppu::Context& c) {
  // sys_mutex_attribute_t: protocol, recursive (0x10 = recursive, 0x20 = not), ...
  const uint32_t attr = A32(1);
  const bool recursive = attr && K().r32(attr + 4) == 0x10;
  uint32_t id = 0;
  mutex_create(recursive, &id);
  K().w32(A32(0), id);
  return CELL_OK;
}

// ---- rwlock (120-127) ----
struct RwLock : Object {
  std::mutex m;
  std::condition_variable cv;
  uint32_t readers = 0;
  uint64_t writer = 0;
  uint32_t writers_waiting = 0;
};
uint64_t sys_rwlock_create(ppu::Context& c) {
  K().w32(A32(0), K().add_object(std::make_shared<RwLock>()));
  return CELL_OK;
}
uint64_t sys_rwlock_rlock(ppu::Context& c, bool try_only) {
  auto l = K().get_object<RwLock>(A32(0));
  if (!l) return CELL_ESRCH;
  std::unique_lock<std::mutex> lk(l->m);
  if (l->writer == me()) return CELL_EDEADLK;
  const auto ready = [&] { return l->writer == 0 && l->writers_waiting == 0; };
  if (try_only) { if (!ready()) return CELL_EBUSY; }
  else if (!wait_for(l->cv, lk, ARG(1), ready)) return CELL_ETIMEDOUT;
  ++l->readers;
  return CELL_OK;
}
uint64_t sys_rwlock_runlock(ppu::Context& c) {
  auto l = K().get_object<RwLock>(A32(0));
  if (!l) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(l->m);
  if (!l->readers) return CELL_EPERM;
  if (--l->readers == 0) l->cv.notify_all();
  return CELL_OK;
}
uint64_t sys_rwlock_wlock(ppu::Context& c, bool try_only) {
  auto l = K().get_object<RwLock>(A32(0));
  if (!l) return CELL_ESRCH;
  std::unique_lock<std::mutex> lk(l->m);
  if (l->writer == me()) return CELL_EDEADLK;
  const auto ready = [&] { return l->writer == 0 && l->readers == 0; };
  if (try_only) { if (!ready()) return CELL_EBUSY; }
  else {
    ++l->writers_waiting;
    const bool ok = wait_for(l->cv, lk, ARG(1), ready);
    --l->writers_waiting;
    if (!ok) { l->cv.notify_all(); return CELL_ETIMEDOUT; }
  }
  l->writer = me();
  return CELL_OK;
}
uint64_t sys_rwlock_wunlock(ppu::Context& c) {
  auto l = K().get_object<RwLock>(A32(0));
  if (!l) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(l->m);
  if (l->writer != me()) return CELL_EPERM;
  l->writer = 0;
  l->cv.notify_all();
  return CELL_OK;
}

// ---- event flags (82-87, 118, 132, 139) ----
struct EventFlag : Object {
  std::mutex m;
  std::condition_variable cv;
  uint64_t bits = 0;
  uint32_t waiters = 0;
  uint32_t cancels = 0;  // generation counter for sys_event_flag_cancel
};
constexpr uint32_t kEfAnd = 0x01, kEfOr = 0x02, kEfClear = 0x10, kEfClearAll = 0x20;
uint64_t sys_event_flag_create(ppu::Context& c) {  // (u32* id, attr*, u64 init)
  auto f = std::make_shared<EventFlag>();
  f->bits = ARG(2);
  K().w32(A32(0), K().add_object(f));
  return CELL_OK;
}
uint64_t sys_event_flag_wait(ppu::Context& c, bool try_only) {  // (id, u64 pattern, mode, u64* result, timeout)
  auto f = K().get_object<EventFlag>(A32(0));
  if (!f) return CELL_ESRCH;
  const uint64_t pat = ARG(1);
  const uint32_t mode = A32(2);
  std::unique_lock<std::mutex> lk(f->m);
  const auto met = [&] { return (mode & kEfAnd) ? (f->bits & pat) == pat : (f->bits & pat) != 0; };
  if (!met()) {
    if (try_only) { K().w64(A32(3), f->bits); return CELL_EBUSY; }
    const uint32_t gen = f->cancels;
    ++f->waiters;
    const bool ok = wait_for(f->cv, lk, ARG(4), [&] { return met() || f->cancels != gen; });
    --f->waiters;
    if (f->cancels != gen) { K().w64(A32(3), f->bits); return CELL_ECANCELED; }
    if (!ok) { K().w64(A32(3), f->bits); return CELL_ETIMEDOUT; }
  }
  K().w64(A32(3), f->bits);
  if (mode & kEfClearAll) f->bits = 0;
  else if (mode & kEfClear) f->bits &= ~pat;
  return CELL_OK;
}
uint64_t sys_event_flag_set(ppu::Context& c) {
  auto f = K().get_object<EventFlag>(A32(0));
  if (!f) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(f->m);
  f->bits |= ARG(1);
  f->cv.notify_all();
  return CELL_OK;
}
uint64_t sys_event_flag_clear(ppu::Context& c) {  // (id, u64 bitptn): bits &= bitptn
  auto f = K().get_object<EventFlag>(A32(0));
  if (!f) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(f->m);
  f->bits &= ARG(1);
  return CELL_OK;
}
uint64_t sys_event_flag_cancel(ppu::Context& c) {  // (id, u32* num)
  auto f = K().get_object<EventFlag>(A32(0));
  if (!f) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(f->m);
  K().w32(A32(1), f->waiters);
  ++f->cancels;
  f->cv.notify_all();
  return CELL_OK;
}
uint64_t sys_event_flag_get(ppu::Context& c) {
  auto f = K().get_object<EventFlag>(A32(0));
  if (!f) { K().w64(A32(1), 0); return CELL_ESRCH; }
  std::lock_guard<std::mutex> lk(f->m);
  K().w64(A32(1), f->bits);
  return CELL_OK;
}

// ---- memory containers (324/325/343/350): sizes only; memory comes from the heap ----
struct MemContainer : Object {
  uint32_t size = 0, used = 0;
};
uint64_t sys_memory_container_create(ppu::Context& c) {  // (u32* cid, size)
  auto m = std::make_shared<MemContainer>();
  m->size = A32(1);
  K().w32(A32(0), K().add_object(m));
  return CELL_OK;
}
uint64_t sys_memory_allocate_from_container(ppu::Context& c) {  // (size, cid, flags, u32* addr)
  auto m = K().get_object<MemContainer>(A32(1));
  if (!m) return CELL_ESRCH;
  const uint32_t align = (ARG(2) & 0x400) ? 0x100000 : 0x10000;
  const uint32_t size = align_up(A32(0), align);
  const uint32_t a = K().heap.alloc(size, align);
  if (!a) return CELL_ENOMEM;
  m->used += size;
  K().w32(A32(3), a);
  return CELL_OK;
}
uint64_t sys_memory_container_get_size(ppu::Context& c) {  // (sys_memory_info_t* {total, available}, cid)
  auto m = K().get_object<MemContainer>(A32(1));
  if (!m) return CELL_ESRCH;
  K().w32(A32(0), m->size);
  K().w32(A32(0) + 4, m->size > m->used ? m->size - m->used : 0);
  return CELL_OK;
}

uint64_t sys_cond_create(ppu::Context& c) {
  uint32_t id = 0;
  const uint32_t rc = cond_create(A32(1), &id);
  if (rc == CELL_OK) K().w32(A32(0), id);
  return rc;
}

uint64_t sys_event_queue_create(ppu::Context& c) {
  const int32_t size = int32_t(ARG(3));
  if (size <= 0 || size > 127) return CELL_EINVAL;
  auto q = std::make_shared<EventQueue>();
  q->size = size;
  q->key = ARG(2);
  K().w32(A32(0), K().add_object(q));
  return CELL_OK;
}
uint64_t sys_event_queue_destroy(ppu::Context& c) {
  auto q = K().get_object<EventQueue>(A32(0));
  if (!q) return CELL_ESRCH;
  {
    std::lock_guard<std::mutex> lk(q->m);
    if (q->waiters && ARG(1) != 1) return CELL_EBUSY;  // mode 1 = force
    q->destroyed = true;
    q->cv.notify_all();
  }
  K().remove_object(A32(0));
  return CELL_OK;
}
// Event is returned in r4..r7.
uint64_t sys_event_queue_receive(ppu::Context& c) {
  auto q = K().get_object<EventQueue>(A32(0));
  if (!q) return CELL_ESRCH;
  std::unique_lock<std::mutex> lk(q->m);
  ++q->waiters;
  const bool ok = wait_for(q->cv, lk, ARG(2), [&] { return !q->events.empty() || q->destroyed; });
  --q->waiters;
  if (q->destroyed) return CELL_ECANCELED;
  if (!ok) return CELL_ETIMEDOUT;
  const Event e = q->events.front();
  q->events.pop_front();
  c.r[4] = e.source; c.r[5] = e.d1; c.r[6] = e.d2; c.r[7] = e.d3;
  return CELL_OK;
}
uint64_t sys_event_queue_tryreceive(ppu::Context& c) {
  auto q = K().get_object<EventQueue>(A32(0));
  if (!q) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(q->m);
  const uint32_t arr = A32(1);
  const int32_t max = int32_t(ARG(2));
  int32_t n = 0;
  while (n < max && !q->events.empty()) {
    const Event e = q->events.front();
    q->events.pop_front();
    K().w64(arr + 32 * n, e.source); K().w64(arr + 32 * n + 8, e.d1);
    K().w64(arr + 32 * n + 16, e.d2); K().w64(arr + 32 * n + 24, e.d3);
    ++n;
  }
  K().w32(A32(3), uint32_t(n));
  return CELL_OK;
}
uint64_t sys_event_queue_drain(ppu::Context& c) {
  auto q = K().get_object<EventQueue>(A32(0));
  if (!q) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(q->m);
  q->events.clear();
  return CELL_OK;
}
uint64_t sys_event_port_create(ppu::Context& c) {
  if (ARG(1) != 1) return CELL_EINVAL;  // only SYS_EVENT_PORT_LOCAL
  auto p = std::make_shared<EventPort>();
  const uint32_t id = K().add_object(p);
  p->name = ARG(2) ? ARG(2) : (0x01000500ull << 32) | id;
  K().w32(A32(0), id);
  return CELL_OK;
}
uint64_t sys_event_port_connect_local(ppu::Context& c) {
  auto p = K().get_object<EventPort>(A32(0));
  auto q = K().get_object<EventQueue>(A32(1));
  if (!p || !q) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(p->m);
  if (p->queue.lock()) return CELL_EISCONN;
  p->queue = q;
  p->queue_id = A32(1);
  return CELL_OK;
}
uint64_t sys_event_port_disconnect(ppu::Context& c) {
  auto p = K().get_object<EventPort>(A32(0));
  if (!p) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(p->m);
  if (!p->queue.lock()) return CELL_ENOTCONN;
  p->queue.reset();
  return CELL_OK;
}
uint64_t sys_event_port_send(ppu::Context& c) {
  auto p = K().get_object<EventPort>(A32(0));
  if (!p) return CELL_ESRCH;
  std::shared_ptr<EventQueue> q;
  uint64_t name;
  {
    std::lock_guard<std::mutex> lk(p->m);
    q = p->queue.lock();
    name = p->name;
  }
  if (!q) return CELL_ENOTCONN;
  std::lock_guard<std::mutex> lk(q->m);
  if (q->destroyed) return CELL_ENOTCONN;
  if (int32_t(q->events.size()) >= q->size) return CELL_EBUSY;
  q->events.push_back(Event{name, ARG(1), ARG(2), ARG(3)});
  q->cv.notify_one();
  return CELL_OK;
}

uint64_t sys_ppu_thread_create_raw(ppu::Context& c) {
  // _sys_ppu_thread_create(u64* id, {u32 entry_opd, u32 tls}* param, u64 arg, u64 unk,
  //                        s32 prio, u32 stacksize, u64 flags, const char* name)
  const uint32_t param = A32(1);
  auto t = K().create_thread(K().r32(param), ARG(2), int32_t(ARG(4)), A32(5), (ARG(6) & 1) != 0,
                             K().read_str(A32(7), 27));
  if (!t) return CELL_ENOMEM;
  K().w64(A32(0), t->id);
  return CELL_OK;
}

uint64_t sys_memory_allocate(ppu::Context& c) {
  const uint32_t align = (ARG(1) & 0x400) ? 0x100000 : 0x10000;
  const uint32_t a = K().heap.alloc(align_up(A32(0), align), align);
  if (!a) return CELL_ENOMEM;
  K().w32(A32(2), a);
  return CELL_OK;
}

// ---- sys_mmapper -------------------------------------------------------------
// Address areas come from [0x40000000, 0xC0000000) in 256 MB-aligned steps (the
// RSX local window sits at 0xC0000000). Memory objects are sizes; mapping one
// commits host pages at that address. (Mapping one object at two addresses does
// not alias them; games use one mapping per object.)
struct Mmapper {
  std::mutex m;
  uint32_t next_area = 0x40000000;
  std::map<uint32_t, uint32_t> areas;     // addr -> size
  std::map<uint32_t, uint32_t> objects;   // mem id -> size
  std::map<uint32_t, uint32_t> mappings;  // addr -> mem id
  uint32_t next_id = 0x08000001;
};
Mmapper& MM() { static Mmapper m; return m; }

uint32_t align_up32(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

}  // namespace

uint32_t mmapper_allocate_address(uint32_t size, uint32_t align, uint32_t* addr) {
  auto& mm = MM();
  std::lock_guard<std::mutex> lk(mm.m);
  if (!size) return CELL_EINVAL;
  align = std::max<uint32_t>(align, 0x10000000);
  const uint32_t a = align_up32(mm.next_area, align);
  const uint64_t end = uint64_t(a) + align_up32(size, 0x10000000);
  if (end > 0xC0000000ull) return CELL_ENOMEM;
  mm.areas[a] = size;
  mm.next_area = uint32_t(end);
  *addr = a;
  return CELL_OK;
}
uint32_t mmapper_allocate_memory(uint32_t size, uint32_t* id) {
  auto& mm = MM();
  std::lock_guard<std::mutex> lk(mm.m);
  if (!size) return CELL_EINVAL;
  *id = mm.next_id++;
  mm.objects[*id] = size;
  return CELL_OK;
}
uint32_t mmapper_free_memory(uint32_t id) {
  auto& mm = MM();
  std::lock_guard<std::mutex> lk(mm.m);
  for (auto& [a, i] : mm.mappings) if (i == id) return CELL_EBUSY;
  return mm.objects.erase(id) ? CELL_OK : CELL_ESRCH;
}
uint32_t mmapper_map_memory(uint32_t addr, uint32_t id) {
  auto& mm = MM();
  std::lock_guard<std::mutex> lk(mm.m);
  auto o = mm.objects.find(id);
  if (o == mm.objects.end()) return CELL_ESRCH;
  if (mm.mappings.count(addr)) return CELL_EBUSY;
  if (!K().memory().map(addr, o->second)) return CELL_ENOMEM;
  mm.mappings[addr] = id;
  return CELL_OK;
}
uint32_t mmapper_search_and_map(uint32_t start, uint32_t id, uint32_t* out) {
  uint32_t size = 0;
  {
    auto& mm = MM();
    std::lock_guard<std::mutex> lk(mm.m);
    auto o = mm.objects.find(id);
    if (o == mm.objects.end()) return CELL_ESRCH;
    size = o->second;
    // first free slot inside the area containing `start`
    auto ar = mm.areas.upper_bound(start);
    if (ar == mm.areas.begin()) return CELL_EINVAL;
    --ar;
    const uint32_t step = size >= 0x100000 ? 0x100000 : 0x10000;
    for (uint64_t a = ar->first; a + size <= uint64_t(ar->first) + ar->second; a += step) {
      bool clash = false;
      for (auto& [ma, mi] : mm.mappings) {
        const uint64_t ms = mm.objects.count(mi) ? mm.objects[mi] : 0;
        if (a < ma + ms && ma < a + size) { clash = true; break; }
      }
      if (!clash) { *out = uint32_t(a); break; }
    }
    if (!*out) return CELL_ENOMEM;
  }
  return mmapper_map_memory(*out, id);
}
uint32_t mmapper_unmap_memory(uint32_t addr, uint32_t* id) {
  auto& mm = MM();
  std::lock_guard<std::mutex> lk(mm.m);
  auto it = mm.mappings.find(addr);
  if (it == mm.mappings.end()) return CELL_EINVAL;
  if (id) *id = it->second;
  mm.mappings.erase(it);  // pages stay committed; contents are simply abandoned
  return CELL_OK;
}

namespace {

uint64_t sys_time_get_current_time(ppu::Context& c) {
  using namespace std::chrono;
  const auto now = system_clock::now().time_since_epoch();
  K().w64(A32(0), uint64_t(duration_cast<seconds>(now).count()));
  K().w64(A32(1), uint64_t(duration_cast<nanoseconds>(now).count() % 1000000000));
  return CELL_OK;
}

uint64_t sys_tty_write(ppu::Context& c) {
  const uint32_t buf = A32(1), len = A32(2);
  std::string s(reinterpret_cast<const char*>(K().base() + buf), len);
  if (K().tty) K().tty(s);
  else std::fwrite(s.data(), 1, s.size(), stdout);
  K().w32(A32(3), len);
  return CELL_OK;
}

}  // namespace

uint32_t event_queue_create(int32_t size, uint64_t key, uint32_t* out_id) {
  if (size <= 0 || size > 127) return CELL_EINVAL;
  auto q = std::make_shared<EventQueue>();
  q->size = size;
  q->key = key;
  *out_id = K().add_object(q);
  return CELL_OK;
}

bool event_queue_send_by_key(uint64_t key, uint64_t source, uint64_t d1, uint64_t d2, uint64_t d3) {
  std::shared_ptr<EventQueue> q;
  K().for_each_object([&](uint32_t, const std::shared_ptr<Object>& o) {
    auto eq = std::dynamic_pointer_cast<EventQueue>(o);
    if (eq && eq->key == key && !q) q = eq;
  });
  if (!q) return false;
  std::lock_guard<std::mutex> lk(q->m);
  if (q->destroyed || int32_t(q->events.size()) >= q->size) return false;
  q->events.push_back(Event{source, d1, d2, d3});
  q->cv.notify_one();
  return true;
}

// Registers plus a guest call trace from the stack back-chain: each PPC64 frame
// holds the caller's r1 at +0 and the saved LR at +16 of the caller's frame.
void Kernel::dump_context(const ppu::Context& c) {
  logf("  pc~lr=0x%08llX ctr=0x%08llX r1=0x%08llX r2=0x%08llX", (unsigned long long)c.lr,
       (unsigned long long)c.ctr, (unsigned long long)c.r[1], (unsigned long long)c.r[2]);
  for (int i = 0; i < 32; i += 4)
    logf("  r%-2d=%016llX r%-2d=%016llX r%-2d=%016llX r%-2d=%016llX", i, (unsigned long long)c.r[i], i + 1,
         (unsigned long long)c.r[i + 1], i + 2, (unsigned long long)c.r[i + 2], i + 3, (unsigned long long)c.r[i + 3]);
  logf("  guest call trace (saved LRs):%s", guest_trace(uint32_t(c.r[1])).c_str());
}

std::string Kernel::guest_trace(uint32_t sp) {
  std::string trace;
  for (int depth = 0; depth < 24 && sp >= kStackBase && sp < kStackBase + kStackSize - 32; ++depth) {
    const uint32_t prev = r32(sp + 4);  // back chain (low word of the 64-bit slot)
    if (prev <= sp || prev >= kStackBase + kStackSize) break;
    char buf[16];
    std::snprintf(buf, sizeof buf, " %08X", r32(prev + 16 + 4));
    trace += buf;
    sp = prev;
  }
  return trace;
}

void Kernel::dump_threads() {
  std::vector<std::shared_ptr<Thread>> ts;
  {
    std::lock_guard<std::mutex> lk(threads_m_);
    for (auto& [id, t] : threads_) ts.push_back(t);
  }
  logf("---- thread dump (%zu PPU threads) ----", ts.size());
  for (auto& t : ts) {
    if (t->finished) continue;
    const uint32_t sys = t->cur_sys.load();
    char what[96];
    if (sys)
      std::snprintf(what, sizeof what, "in %u %s(r3=0x%X) from 0x%08X", sys & 0x7FFFFFFF, syscall_name(sys & 0x7FFFFFFF),
                    t->cur_sys_r3.load(), t->cur_sys_lr.load());
    else
      std::snprintf(what, sizeof what, "running, lr=0x%08X", uint32_t(t->ctx.lr));
    logf("  %-24s %s", t->name.c_str(), what);
    logf("     trace:%s", guest_trace(uint32_t(t->ctx.r[1])).c_str());
  }
  for_each_object([&](uint32_t id, const std::shared_ptr<Object>& o) {
    if (auto* s = dynamic_cast<SpuThread*>(o.get()))
      logf("  spu 0x%X %-20s pc=0x%05X in_mbox=%zu out_mbox=%zu", id, s->name.c_str(), s->core.pc, s->in_mbox.size(),
           s->out_mbox.size());
  });
}

void Kernel::start_thread_dumper() {
  const char* s = std::getenv("PS3RECOMP_THREAD_DUMP");
  if (!s) return;
  const int secs = std::max(1, std::atoi(s));
  std::thread([this, secs] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(secs));
      dump_threads();
    }
  }).detach();
}

uint32_t event_queue_send(uint32_t id, uint64_t source, uint64_t d1, uint64_t d2, uint64_t d3) {
  auto q = K().get_object<EventQueue>(id);
  if (!q) return CELL_ESRCH;
  std::lock_guard<std::mutex> lk(q->m);
  if (q->destroyed) return CELL_ESRCH;
  if (int32_t(q->events.size()) >= q->size) return CELL_EBUSY;
  q->events.push_back(Event{source, d1, d2, d3});
  q->cv.notify_one();
  return CELL_OK;
}

uint32_t event_queue_receive(uint32_t id, uint64_t out[4], bool wait) {
  auto q = K().get_object<EventQueue>(id);
  if (!q) return CELL_ESRCH;
  std::unique_lock<std::mutex> lk(q->m);
  if (wait) {
    ++q->waiters;
    q->cv.wait(lk, [&] { return !q->events.empty() || q->destroyed; });
    --q->waiters;
  }
  if (q->destroyed) return CELL_ECANCELED;
  if (q->events.empty()) return CELL_EBUSY;
  const Event e = q->events.front();
  q->events.pop_front();
  out[0] = e.source; out[1] = e.d1; out[2] = e.d2; out[3] = e.d3;
  return CELL_OK;
}

const char* syscall_name(uint32_t n) {
  switch (n) {
    case 1: return "sys_process_getpid";
    case 22: return "sys_process_exit";
    case 25: return "sys_process_get_sdk_version";
    case 41: return "_sys_ppu_thread_exit";
    case 43: return "sys_ppu_thread_yield";
    case 44: return "sys_ppu_thread_join";
    case 45: return "sys_ppu_thread_detach";
    case 46: return "sys_ppu_thread_get_join_state";
    case 47: return "sys_ppu_thread_set_priority";
    case 48: return "sys_ppu_thread_get_priority";
    case 49: return "sys_ppu_thread_get_stack_information";
    case 52: return "_sys_ppu_thread_create";
    case 53: return "sys_ppu_thread_start";
    case 82: return "sys_event_flag_create";
    case 90: return "sys_semaphore_create";
    case 91: return "sys_semaphore_destroy";
    case 92: return "sys_semaphore_wait";
    case 93: return "sys_semaphore_trywait";
    case 94: return "sys_semaphore_post";
    case 100: return "sys_mutex_create";
    case 101: return "sys_mutex_destroy";
    case 102: return "sys_mutex_lock";
    case 103: return "sys_mutex_trylock";
    case 104: return "sys_mutex_unlock";
    case 105: return "sys_cond_create";
    case 106: return "sys_cond_destroy";
    case 107: return "sys_cond_wait";
    case 108: return "sys_cond_signal";
    case 109: return "sys_cond_signal_all";
    case 110: return "sys_cond_signal_to";
    case 114: return "sys_semaphore_get_value";
    case 128: return "sys_event_queue_create";
    case 129: return "sys_event_queue_destroy";
    case 130: return "sys_event_queue_receive";
    case 131: return "sys_event_queue_tryreceive";
    case 133: return "sys_event_queue_drain";
    case 134: return "sys_event_port_create";
    case 135: return "sys_event_port_destroy";
    case 136: return "sys_event_port_connect_local";
    case 137: return "sys_event_port_disconnect";
    case 138: return "sys_event_port_send";
    case 141: return "sys_timer_usleep";
    case 142: return "sys_timer_sleep";
    case 144: return "sys_time_get_timezone";
    case 145: return "sys_time_get_current_time";
    case 147: return "sys_time_get_timebase_frequency";
    case 300: return "sys_vm_memory_map";
    case 301: return "sys_vm_unmap";
    case 302: return "sys_vm_append_memory";
    case 303: return "sys_vm_return_memory";
    case 311: return "sys_vm_test";
    case 312: return "sys_vm_get_statistics";
    case 329: return "sys_mmapper_free_shared_memory";
    case 330: return "sys_mmapper_allocate_address";
    case 331: return "sys_mmapper_free_address";
    case 332: return "sys_mmapper_allocate_shared_memory";
    case 334: return "sys_mmapper_map_shared_memory";
    case 335: return "sys_mmapper_unmap_shared_memory";
    case 337: return "sys_mmapper_search_and_map";
    case 85: return "sys_event_flag_wait";
    case 87: return "sys_event_flag_set";
    case 118: return "sys_event_flag_clear";
    case 120: return "sys_rwlock_create";
    case 122: return "sys_rwlock_rlock";
    case 124: return "sys_rwlock_runlock";
    case 125: return "sys_rwlock_wlock";
    case 127: return "sys_rwlock_wunlock";
    case 324: return "sys_memory_container_create";
    case 350: return "sys_memory_allocate_from_container";
    case 169: return "sys_spu_initialize";
    case 170: return "sys_spu_thread_group_create";
    case 171: return "sys_spu_thread_group_destroy";
    case 172: return "sys_spu_thread_initialize";
    case 173: return "sys_spu_thread_group_start";
    case 177: return "sys_spu_thread_group_terminate";
    case 178: return "sys_spu_thread_group_join";
    case 181: return "sys_spu_thread_write_ls";
    case 182: return "sys_spu_thread_read_ls";
    case 184: return "sys_spu_thread_write_snr";
    case 185: return "sys_spu_thread_group_connect_event";
    case 190: return "sys_spu_thread_write_spu_mb";
    case 191: return "sys_spu_thread_connect_event";
    case 193: return "sys_spu_thread_bind_queue";
    case 251: return "sys_spu_thread_group_connect_event_all_threads";
    case 348: return "sys_memory_allocate";
    case 349: return "sys_memory_free";
    case 352: return "sys_memory_get_user_memory_size";
    case 403: return "sys_tty_write";
    default: return "";
  }
}

void Kernel::syscall(ppu::Context& c) {
  const uint32_t n = uint32_t(c.r[11]);
  // Profiling: time spent in each syscall per guest call site.
  struct CallTimer {
    uint32_t n, lr;
    Clock::time_point t0;
    bool on;
    ~CallTimer() {
      if (on)
        prof::record_call(n, lr, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count()),
                          Kernel::get().current().name);
    }
  } timer{n, uint32_t(c.lr), Clock::now(), prof::enabled()};
  struct SysMark {
    Thread& t;
    ~SysMark() { t.cur_sys = 0; }
  } mark{current()};
  mark.t.cur_sys_r3 = uint32_t(c.r[3]);
  mark.t.cur_sys_lr = uint32_t(c.lr);
  mark.t.cur_sys = n | 0x80000000u;
  if (n == 141 && prof::enabled()) {  // which sleep lengths does each caller ask for?
    static std::mutex m;
    static std::map<uint32_t, int> seen;
    std::lock_guard<std::mutex> lk(m);
    if (seen[uint32_t(c.lr)]++ < 3) logf("usleep(%llu) from 0x%08X", (unsigned long long)c.r[3], uint32_t(c.lr));
  }
  static const bool trace = std::getenv("PS3RECOMP_SYSCALL_TRACE") != nullptr;
  static std::atomic<uint32_t> traced{0};
  if (trace && traced++ < 400) {
    static const auto t0 = Clock::now();
    logf("syscall %3u %-28s r3=0x%llX r4=0x%llX r5=0x%llX lr=0x%llX  t=%.1fms", n, syscall_name(n),
         (unsigned long long)c.r[3], (unsigned long long)c.r[4], (unsigned long long)c.r[5],
         (unsigned long long)c.lr, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  // Debug watch: PS3RECOMP_WATCH=<hex guest address> logs when that word changes,
  // checked at every syscall (coarse, but enough to find the culprit's neighbourhood).
  static const uint32_t watch = [] {
    const char* w = std::getenv("PS3RECOMP_WATCH");
    return w ? uint32_t(std::strtoul(w, nullptr, 16)) : 0u;
  }();
#if defined(_WIN32)
  // Only once the watched page is committed (it may belong to a later allocation).
  const bool watch_mapped = watch && [&] {
    MEMORY_BASIC_INFORMATION mbi{};
    return VirtualQuery(base() + watch, &mbi, sizeof mbi) && mbi.State == MEM_COMMIT;
  }();
#else
  const bool watch_mapped = watch != 0;
#endif
  if (watch_mapped) {
    static std::atomic<uint32_t> last{0xDEADBEEF};
    const uint32_t v = r32(watch);
    const uint32_t old = last.exchange(v);
    if (old != v) logf("watch [0x%08X] = 0x%08X (was 0x%08X) before syscall %u, lr=0x%llX", watch, v, old, n, (unsigned long long)c.lr);
  }
  uint64_t rc = CELL_OK;
  switch (n) {
    case 1: rc = 0x01000500; break;
    case 3: case 22: case 26: throw ProcessExit{int(int32_t(ARG(0)))};  // exit3 / _sys_process_exit / exit2
    case 25: w32(A32(1), sdk_version); break;

    case 41: throw ThreadExit{ARG(0)};
    case 43: std::this_thread::yield(); break;
    case 44: {
      uint64_t v = 0;
      rc = join_thread(ARG(0), &v);
      if (rc == CELL_OK) w64(A32(1), v);
      break;
    }
    case 45: rc = detach_thread(ARG(0)); break;
    case 46: w32(A32(0), current().joinable ? 1 : 0); break;
    case 47:
      if (auto t = find_thread(ARG(0))) {
        t->prio = int32_t(ARG(1));
        if (t.get() == t_current) apply_host_priority(t->prio);
      } else {
        rc = CELL_ESRCH;
      }
      break;
    case 48:
      if (auto t = find_thread(ARG(0))) w32(A32(1), uint32_t(t->prio)); else rc = CELL_ESRCH;
      break;
    case 49: {
      Thread& t = current();
      w32(A32(0), t.stack_addr);
      w32(A32(0) + 4, t.stack_size);
      break;
    }
    case 52: rc = sys_ppu_thread_create_raw(c); break;
    case 53: rc = start_thread(ARG(0)); break;

    case 90: rc = sys_semaphore_create(c); break;
    case 91: rc = remove_object(A32(0)) ? CELL_OK : CELL_ESRCH; break;
    case 92: rc = sys_semaphore_wait(c, false); break;
    case 93: rc = sys_semaphore_wait(c, true); break;
    case 94: rc = sys_semaphore_post(c); break;
    case 114: rc = sys_semaphore_get_value(c); break;

    case 100: rc = sys_mutex_create(c); break;
    case 101: rc = mutex_destroy(A32(0)); break;
    case 102: rc = mutex_lock(A32(0), ARG(1)); break;
    case 103: rc = mutex_trylock(A32(0)); break;
    case 104: rc = mutex_unlock(A32(0)); break;

    case 105: rc = sys_cond_create(c); break;
    case 106: rc = cond_destroy(A32(0)); break;
    case 107: rc = cond_wait(A32(0), ARG(1)); break;
    case 108: rc = cond_signal(A32(0)); break;
    case 109: rc = cond_signal_all(A32(0)); break;
    case 110: rc = cond_signal_to(A32(0), ARG(1)); break;

    case 82: rc = sys_event_flag_create(c); break;
    case 83: rc = remove_object(A32(0)) ? CELL_OK : CELL_ESRCH; break;
    case 85: rc = sys_event_flag_wait(c, false); break;
    case 86: rc = sys_event_flag_wait(c, true); break;
    case 87: rc = sys_event_flag_set(c); break;
    case 118: rc = sys_event_flag_clear(c); break;
    case 132: rc = sys_event_flag_cancel(c); break;
    case 139: rc = sys_event_flag_get(c); break;

    case 120: rc = sys_rwlock_create(c); break;
    case 121: rc = remove_object(A32(0)) ? CELL_OK : CELL_ESRCH; break;
    case 122: rc = sys_rwlock_rlock(c, false); break;
    case 123: rc = sys_rwlock_rlock(c, true); break;
    case 124: rc = sys_rwlock_runlock(c); break;
    case 125: rc = sys_rwlock_wlock(c, false); break;
    case 126: rc = sys_rwlock_wlock(c, true); break;
    case 127: rc = sys_rwlock_wunlock(c); break;

    case 324: rc = sys_memory_container_create(c); break;
    case 325: rc = remove_object(A32(0)) ? CELL_OK : CELL_ESRCH; break;
    case 343: rc = sys_memory_container_get_size(c); break;
    case 350: rc = sys_memory_allocate_from_container(c); break;

    case 128: rc = sys_event_queue_create(c); break;
    case 129: rc = sys_event_queue_destroy(c); break;
    case 130: rc = sys_event_queue_receive(c); break;
    case 131: rc = sys_event_queue_tryreceive(c); break;
    case 133: rc = sys_event_queue_drain(c); break;
    case 134: rc = sys_event_port_create(c); break;
    case 135: rc = remove_object(A32(0)) ? CELL_OK : CELL_ESRCH; break;
    case 136: rc = sys_event_port_connect_local(c); break;
    case 137: rc = sys_event_port_disconnect(c); break;
    case 138: rc = sys_event_port_send(c); break;

    case 141: precise_sleep(ARG(0)); break;
    case 142: precise_sleep(uint64_t(A32(0)) * 1000000); break;
    case 144: w32(A32(0), 0); w32(A32(1), 0); break;  // sys_time_get_timezone
    case 145: rc = sys_time_get_current_time(c); break;
    case 147: rc = kTimebaseFrequency; break;

    // sys_vm: a virtual area backed by host memory committed up front, so the
    // append/return/lock/touch/flush family has nothing left to do.
    case 300: {  // sys_vm_memory_map(vsize, psize, container, flag, policy, u32* addr)
      uint32_t a = 0;
      rc = mmapper_allocate_address(A32(0), 0x10000000, &a);
      if (rc == CELL_OK && !memory().map(a, A32(0))) rc = CELL_ENOMEM;
      if (rc == CELL_OK) {
        w32(A32(5), a);
        logf("sys_vm_memory_map: %u MB virtual at 0x%08X", A32(0) >> 20, a);
      }
      break;
    }
    case 301: case 302: case 303: case 304: case 305: case 306: case 307: case 308: case 309: case 310:
      break;  // unmap / append / return / lock / unlock / touch / flush / invalidate / store / sync
    case 311: w64(A32(2), 0); break;  // sys_vm_test(addr, size, u64* result)
    case 312: {  // sys_vm_get_statistics(addr, stat*): zeroed counters
      if (A32(1)) std::memset(base() + A32(1), 0, 32);
      break;
    }
    // sys_mmapper
    case 329: rc = mmapper_free_memory(A32(0)); break;
    case 330: {  // allocate_address(size, flags, alignment, u32* addr)
      uint32_t a = 0;
      rc = mmapper_allocate_address(A32(0), A32(2), &a);
      if (rc == CELL_OK) w32(A32(3), a);
      break;
    }
    case 331: break;  // free_address: areas are not reused
    case 332: {  // allocate_shared_memory(u64 key, size, flags, u32* id)
      uint32_t id = 0;
      rc = mmapper_allocate_memory(A32(1), &id);
      if (rc == CELL_OK) w32(A32(3), id);
      break;
    }
    case 334: rc = mmapper_map_memory(A32(0), A32(1)); break;
    case 335: {
      uint32_t id = 0;
      rc = mmapper_unmap_memory(A32(0), &id);
      if (rc == CELL_OK) w32(A32(1), id);
      break;
    }
    case 337: {  // search_and_map(start, id, flags, u32* addr)
      uint32_t a = 0;
      rc = mmapper_search_and_map(A32(0), A32(1), &a);
      if (rc == CELL_OK) w32(A32(3), a);
      break;
    }
    case 348: rc = sys_memory_allocate(c); break;
    case 349: rc = heap.free(A32(0)) ? CELL_OK : CELL_EINVAL; break;
    case 352: w32(A32(0), heap.total()); w32(A32(0) + 4, heap.total() - heap.used()); break;

    case 403: rc = sys_tty_write(c); break;
    case 988: break;  // debug-build PPU exception handler registration: accepted, nothing to do

    default: {
      if (spu_syscall(n, c, rc)) break;
      static std::mutex seen_m;
      static std::set<uint32_t> seen;
      bool first;
      {
        std::lock_guard<std::mutex> lk(seen_m);
        first = seen.insert(n).second;
      }
      if (strict) throw std::runtime_error("unimplemented syscall " + std::to_string(n));
      if (first) logf("unimplemented syscall %u %s (returning CELL_OK)", n, syscall_name(n));
      break;
    }
  }
  c.r[3] = rc;
  // Errors other than the usual timeouts/busy polls, once per (syscall, call site).
  if ((uint32_t(rc) & 0xFFFF0000u) == 0x80010000u && uint32_t(rc) != 0x8001000B /*ETIMEDOUT*/ &&
      uint32_t(rc) != 0x8001000A /*EBUSY*/) {
    static std::mutex m;
    static std::set<std::pair<uint32_t, uint32_t>> seen;
    std::lock_guard<std::mutex> lk(m);
    if (seen.insert({n, uint32_t(c.lr)}).second && seen.size() < 200)
      logf("syscall %u %s -> 0x%08X from 0x%08X [%s]", n, syscall_name(n), uint32_t(rc), uint32_t(c.lr), current().name.c_str());
  }
}

#undef ARG
#undef A32

}  // namespace lv2
