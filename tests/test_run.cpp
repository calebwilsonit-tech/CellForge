// Runs the recompiled test functions and checks their results.
// usage: ps3_test_run <test.elf> <mods dir>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "elf64.h"
#include "guest_memory.h"
#include "ppu_mod_api.h"
#include "ppu_runtime.h"
#include "test_layout.h"

// Every VMX helper named in the instruction table must exist (compile-time check).
#define VOP(e, n, k, form, xo) [[maybe_unused]] static const void* const vop_ref_##n = reinterpret_cast<const void*>(&ppu::vmx::n);
#define mfvscr cr6_from
#define mtvscr cr6_from
#include "ppu_vmx_ops.inc"
#undef mfvscr
#undef mtvscr
#undef VOP

namespace {

using namespace test_layout;

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* what, unsigned long long got, unsigned long long want) {
  ++g_checks;
  if (ok) {
    std::printf("  ok    %s\n", what);
  } else {
    std::printf("  FAIL  %s: got 0x%llX, want 0x%llX\n", what, got, want);
    ++g_failures;
  }
}
#define CHECK_EQ(expr, want, what) do { const unsigned long long g_ = (expr); check(g_ == (unsigned long long)(want), what, g_, (unsigned long long)(want)); } while (0)

constexpr uint32_t kStackBase = 0x00200000;
constexpr uint32_t kStackSize = 0x00100000;
constexpr uint32_t kStackTop = kStackBase + kStackSize - 0x100;
constexpr uint32_t kScratch = 0x00300000;

ppu::GuestMemory* g_mem = nullptr;

ppu::Context run(uint32_t addr, uint64_t r3 = 0, uint64_t r4 = 0) {
  ppu::Context c;
  c.base = g_mem->base();
  c.r[1] = kStackTop;
  c.r[2] = kToc;
  c.r[3] = r3;
  c.r[4] = r4;
  ppu::call_indirect(c, addr);
  return c;
}

void be_f32(uint32_t ea, float v) { ppu::st32(g_mem->base(), ea, std::bit_cast<uint32_t>(v)); }
float rd_f32(uint32_t ea) { return std::bit_cast<float>(ppu::ld32(g_mem->base(), ea)); }

void sum_hook(ppu::Context& c) { c.r[3] = 99; }
void log_line(const char* s) { std::printf("  %s\n", s); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: ps3_test_run <test.elf> <mods dir>\n"); return 2; }

  elf::LoadResult lr = elf::load_file(argv[1]);
  if (!lr.ok) { std::fprintf(stderr, "load error: %s\n", lr.error.c_str()); return 1; }

  ppu::GuestMemory mem;
  g_mem = &mem;
  if (!mem.ok()) { std::fprintf(stderr, "cannot reserve guest address space\n"); return 1; }
  for (const auto& s : lr.image.segments) {
    if (!mem.map(uint32_t(s.vaddr), s.memsz)) { std::fprintf(stderr, "map failed\n"); return 1; }
    std::memcpy(mem.base() + s.vaddr, s.data.data(), s.data.size());
  }
  mem.map(kStackBase, kStackSize);
  mem.map(kScratch, 0x1000);
  uint8_t* const base = mem.base();

  ppu_register_functions();

  try {
    std::printf("control flow\n");
    CHECK_EQ(run(k_sum, 10).r[3], 55, "sum(10) == 55 (loop, compare, bc)");
    CHECK_EQ(run(k_sum, 0).r[3], 0, "sum(0) == 0");
    {
      ppu::Context c = run(k_call_sum);
      CHECK_EQ(c.r[3], 55, "call_sum() == 55 (bl)");
      CHECK_EQ(c.r[1], kStackTop, "stack pointer restored");
    }
    CHECK_EQ(run(k_call_indirect).r[3], 28, "call_indirect() == 28 (bctrl)");

    std::printf("jump tables\n");
    for (int k = 0; k < 4; ++k) CHECK_EQ(run(k_sw_abs, k).r[3], 100 + k, "sw_abs(k) absolute table");
    CHECK_EQ(run(k_sw_abs, 9).r[3], ~0ull, "sw_abs(9) default");
    for (int k = 0; k < 3; ++k) CHECK_EQ(run(k_sw_rel, k).r[3], 200 + k, "sw_rel(k) TOC-relative table");
    CHECK_EQ(run(k_sw_rel, 7).r[3], ~0ull, "sw_rel(7) default");

    std::printf("integer\n");
    CHECK_EQ(run(k_shl2, 5).r[3], 20, "rlwinm: 5 << 2");
    CHECK_EQ(run(k_shl2, 0x40000001).r[3], 4, "rlwinm masks wrapped bits");
    {
      ppu::Context c = run(k_carry, ~0ull, 1);
      CHECK_EQ(c.r[3], 1, "addc/addze carry out");
      CHECK_EQ(c.r[4], 0, "addc wraps");
      c = run(k_carry, 5, 6);
      CHECK_EQ(c.r[3], 0, "addc no carry");
      CHECK_EQ(c.r[4], 11, "addc sum");
    }
    {
      ppu::Context c = run(k_sra, uint64_t(-32));
      CHECK_EQ(c.r[3], uint64_t(-2), "srawi -32 >> 4");
      CHECK_EQ(c.r[4], 0, "srawi CA clear when no 1-bits lost");
      c = run(k_sra, uint64_t(-33));
      CHECK_EQ(c.r[3], uint64_t(-3), "srawi -33 >> 4");
      CHECK_EQ(c.r[4], 1, "srawi CA set");
    }

    std::printf("memory / atomics\n");
    {
      ppu::Context c = run(k_store_load, 0x11223344, kScratch);
      CHECK_EQ(c.r[3], 0x11, "lbz sees most significant byte first");
      CHECK_EQ(base[kScratch + 3], 0x44, "guest bytes are big-endian");
      ppu::st32(base, kScratch + 0x10, 41);
      c = run(k_atomic_inc, kScratch + 0x10);
      CHECK_EQ(c.r[3], 42, "lwarx/stwcx. increment result");
      CHECK_EQ(ppu::ld32(base, kScratch + 0x10), 42, "lwarx/stwcx. stored");
    }

    std::printf("floating point\n");
    {
      ppu::st64(base, kScratch + 0x100, std::bit_cast<uint64_t>(2.5));
      ppu::st64(base, kScratch + 0x108, std::bit_cast<uint64_t>(4.0));
      ppu::Context c = run(k_fp, kScratch + 0x100);
      CHECK_EQ(c.r[3], 12, "fmadd + fctiwz: trunc(2.5*4+2.5)");
      CHECK_EQ(c.r[4], 1, "fcmpu sets cr1.lt");
    }

    std::printf("vector\n");
    {
      const uint32_t p = kScratch + 0x200;
      for (int i = 0; i < 4; ++i) { be_f32(p + 4 * i, float(i + 1)); be_f32(p + 16 + 4 * i, float(10 * (i + 1))); }
      ppu::Context c = run(k_vmx, p);
      bool add_ok = true, splat_ok = true, perm_ok = true;
      for (int i = 0; i < 4; ++i) {
        add_ok &= rd_f32(p + 32 + 4 * i) == float(11 * (i + 1));
        splat_ok &= rd_f32(p + 48 + 4 * i) == 3.0f;
      }
      for (int i = 0; i < 16; ++i) perm_ok &= base[p + 64 + i] == base[p + 1 + i];
      CHECK_EQ(add_ok, 1, "lvx + vaddfp + stvx");
      CHECK_EQ(splat_ok, 1, "vspltw");
      CHECK_EQ(perm_ok, 1, "lvsl + vperm unaligned load");
      CHECK_EQ(c.r[3], 1, "vcmpgtfp. sets cr6 all-true");
    }

    std::printf("discovery\n");
    CHECK_EQ(ppu::find_function(k_opd_only) != nullptr, 1, "function reachable only via OPD was recompiled");
    CHECK_EQ(run(k_opd_only).r[3], 77, "opd_only() == 77");

    std::printf("hooks\n");
    {
      ppu::Fn orig = ppu::hook_function(k_sum, sum_hook);
      CHECK_EQ(orig != nullptr, 1, "hook_function returns the original");
      CHECK_EQ(run(k_call_sum).r[3], 99, "direct bl call goes through the hook");
      CHECK_EQ(run(k_call_indirect).r[3], 99, "indirect bctrl call goes through the hook");
      ppu::unhook_function(k_sum);
      CHECK_EQ(run(k_call_sum).r[3], 55, "unhook restores the original");
    }

    std::printf("mods\n");
    {
      const int n = ppu::load_mods(argv[2], base, log_line);
      CHECK_EQ(n, 1, "example_mod loaded");
      CHECK_EQ(run(k_shl2, 5).r[3], 21, "mod hook chains to the original (20 + 1)");
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  exception: %s\n", e.what());
    ++g_failures;
  }

  if (g_failures) std::printf("%d of %d checks FAILED\n", g_failures, g_checks);
  else std::printf("all %d checks passed\n", g_checks);
  return g_failures ? 1 : 0;
}
