#pragma once
// Built-in sampling profiler (Windows). With PS3RECOMP_PROFILE=<file>, a sampler
// thread suspends every registered thread about once per millisecond, records its
// instruction pointer (as an offset into the executable) and rewrites <file> every
// few seconds with per-thread CPU time and the hottest offsets. Offsets are
// resolved to functions with the linker map (tools/profile_report.py).

#include <cstdint>
#include <string>

namespace prof {

// Call on the thread itself, once, at its start.
void register_thread(const std::string& name);
// Optionally, right after register_thread: a guest program counter the thread keeps
// (roughly) current, sampled alongside the host IP (SPU local-store addresses).
void register_guest_pc(const uint32_t* pc, const uint8_t* ls = nullptr, uint32_t ls_size = 0);

// True when profiling is on (PS3RECOMP_PROFILE set).
bool enabled();
// Records one blocking call (syscall number or HLE id, guest caller) and its duration.
void record_call(uint32_t id, uint32_t caller, uint64_t micros, const std::string& thread);

}  // namespace prof
