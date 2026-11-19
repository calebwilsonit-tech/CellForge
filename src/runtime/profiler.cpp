#define _CRT_SECURE_NO_WARNINGS
#include "profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace prof {
namespace {

#if defined(_WIN32)
struct Entry {
  std::string name;
  HANDLE h = nullptr;
  uint64_t samples = 0;
  std::unordered_map<uint64_t, uint32_t> hits;  // ip offset -> count
  const uint32_t* guest_pc = nullptr;
  std::unordered_map<uint32_t, uint32_t> guest_hits;
  const uint8_t* ls = nullptr;  // guest local store, dumped once for offline disassembly
  uint32_t ls_size = 0;
  bool ls_dumped = false;
};

std::mutex g_m;
std::vector<Entry*> g_threads;
const char* g_file = nullptr;

struct CallStat { uint64_t count = 0, micros = 0; };
std::mutex g_calls_m;
std::map<std::tuple<std::string, uint32_t, uint32_t>, CallStat> g_calls;  // (thread, id, caller) -> stats

uint64_t cpu_ms(HANDLE h) {
  FILETIME c, e, k, u;
  if (!GetThreadTimes(h, &c, &e, &k, &u)) return 0;
  const uint64_t kk = (uint64_t(k.dwHighDateTime) << 32) | k.dwLowDateTime;
  const uint64_t uu = (uint64_t(u.dwHighDateTime) << 32) | u.dwLowDateTime;
  return (kk + uu) / 10000;
}

void write_report() {
  FILE* f = std::fopen(g_file, "w");
  if (!f) return;
  std::lock_guard<std::mutex> lk(g_m);
  for (Entry* e : g_threads) {
    std::fprintf(f, "thread %s cpu_ms %llu samples %llu\n", e->name.c_str(), (unsigned long long)cpu_ms(e->h),
                 (unsigned long long)e->samples);
    // Samples outside the executable are attributed to their module.
    const uintptr_t module = uintptr_t(GetModuleHandleW(nullptr));
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(module + dos->e_lfanew);
    const uint64_t image_size = nt->OptionalHeader.SizeOfImage;
    std::unordered_map<uint64_t, uint32_t> in_exe;
    std::map<std::string, uint32_t> external;
    for (auto& [off, n] : e->hits) {
      if (off < image_size) { in_exe[off] += n; continue; }
      HMODULE m = nullptr;
      char name[MAX_PATH] = "?";
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(module + off), &m))
        GetModuleFileNameA(m, name, sizeof name);
      const char* base = std::strrchr(name, '\\');
      external[base ? base + 1 : name] += n;
    }
    for (auto& [mod, n] : external) std::fprintf(f, "  module %s %u\n", mod.c_str(), n);
    std::vector<std::pair<uint64_t, uint32_t>> v(in_exe.begin(), in_exe.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < v.size() && i < 400; ++i)
      std::fprintf(f, "  %llX %u\n", (unsigned long long)v[i].first, v[i].second);
    std::vector<std::pair<uint32_t, uint32_t>> g(e->guest_hits.begin(), e->guest_hits.end());
    std::sort(g.begin(), g.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < g.size() && i < 60; ++i) std::fprintf(f, "  guest %05X %u\n", g[i].first, g[i].second);
    if (e->ls && !e->ls_dumped && e->samples > 20000) {
      // <profile>.<thread>.ls: raw local store, once the thread has run ~20 s.
      std::string path = std::string(g_file) + "." + e->name + ".ls";
      for (size_t i = std::strlen(g_file); i < path.size(); ++i)
        if (path[i] == ':' || path[i] == ' ' || path[i] == '#' || path[i] == '\\' || path[i] == '/') path[i] = '_';
      if (FILE* d = std::fopen(path.c_str(), "wb")) {
        std::fwrite(e->ls, 1, e->ls_size, d);
        std::fclose(d);
      }
      e->ls_dumped = true;
    }
  }
  {
    std::lock_guard<std::mutex> lc(g_calls_m);
    std::vector<std::pair<std::tuple<std::string, uint32_t, uint32_t>, CallStat>> v(g_calls.begin(), g_calls.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.micros > b.second.micros; });
    for (size_t i = 0; i < v.size() && i < 120; ++i)
      std::fprintf(f, "call %u caller %08X count %llu micros %llu thread %s\n", std::get<1>(v[i].first), std::get<2>(v[i].first),
                   (unsigned long long)v[i].second.count, (unsigned long long)v[i].second.micros, std::get<0>(v[i].first).c_str());
  }
  std::fclose(f);
}

// PS3RECOMP_PROFILE_START=<seconds>: ignore everything before this point.
bool started() {
  static const auto t0 = std::chrono::steady_clock::now();
  static const double delay = [] {
    const char* s = std::getenv("PS3RECOMP_PROFILE_START");
    return s ? std::atof(s) : 0.0;
  }();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= delay;
}

void sampler() {
  const uintptr_t module = uintptr_t(GetModuleHandleW(nullptr));
  auto last = std::chrono::steady_clock::now();
  for (;;) {
    Sleep(1);
    if (!started()) continue;
    {
      std::lock_guard<std::mutex> lk(g_m);
      for (Entry* e : g_threads) {
        if (SuspendThread(e->h) == DWORD(-1)) continue;
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_CONTROL;
        const bool got = GetThreadContext(e->h, &ctx) != 0;
        ResumeThread(e->h);  // before touching the heap: the target may hold its lock
        if (got) {
          ++e->samples;
          ++e->hits[uint64_t(ctx.Rip - module)];
          if (e->guest_pc) ++e->guest_hits[uint32_t(*static_cast<const volatile uint32_t*>(e->guest_pc))];
        }
      }
    }
    if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
      write_report();
      last = std::chrono::steady_clock::now();
    }
  }
}
#endif

}  // namespace

bool enabled() {
  static const bool on = std::getenv("PS3RECOMP_PROFILE") != nullptr;
  return on;
}

void record_call(uint32_t id, uint32_t caller, uint64_t micros, const std::string& thread) {
#if defined(_WIN32)
  if (!started()) return;
  std::lock_guard<std::mutex> lk(g_calls_m);
  CallStat& s = g_calls[{thread, id, caller}];
  ++s.count;
  s.micros += micros;
#else
  (void)id; (void)caller; (void)micros; (void)thread;
#endif
}

void register_thread(const std::string& name) {
#if defined(_WIN32)
  static const char* file = std::getenv("PS3RECOMP_PROFILE");
  if (!file) return;
  std::lock_guard<std::mutex> lk(g_m);
  if (!g_file) {
    g_file = file;
    std::thread(sampler).detach();
  }
  auto* e = new Entry;
  e->name = name;
  DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &e->h,
                  THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
  g_threads.push_back(e);
#else
  (void)name;
#endif
}

void register_guest_pc(const uint32_t* pc, const uint8_t* ls, uint32_t ls_size) {
#if defined(_WIN32)
  if (!enabled()) return;
  std::lock_guard<std::mutex> lk(g_m);
  // The calling thread registered itself last among threads with this handle.
  const DWORD me = GetCurrentThreadId();
  for (auto it = g_threads.rbegin(); it != g_threads.rend(); ++it)
    if (GetThreadId((*it)->h) == me) {
      (*it)->guest_pc = pc;
      (*it)->ls = ls;
      (*it)->ls_size = ls_size;
      return;
    }
#else
  (void)pc; (void)ls; (void)ls_size;
#endif
}

}  // namespace prof
