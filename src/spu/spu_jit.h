#pragma once
// x86-64 AVX2 basic-block JIT for the Cell SPU.
// Active on Windows x64; other platforms fall back to the interpreter.

#include <cstdint>
#include <unordered_map>

namespace spu {

class SpuCore;

class SpuJit {
 public:
  SpuJit();
  ~SpuJit();
  SpuJit(const SpuJit&) = delete;
  SpuJit& operator=(const SpuJit&) = delete;

  bool available() const { return code_ != nullptr; }

  // Runs the SPU from core.pc until a final `stop` or exit_requested().
  uint32_t run(SpuCore& core);

  // Invalidates compiled blocks if [addr, addr+len) overlaps any compiled 4KB page.
  void invalidate_ls(uint32_t addr, uint32_t len);
  void invalidate_all();

 private:
  using BlockFn = void*;
  using EnterFn = uint32_t (*)(SpuCore* self, void* gpr, uint8_t* ls, BlockFn block);

  void build_trampoline();
  BlockFn compile_block(SpuCore& core, uint32_t start_pc);

  uint8_t* code_ = nullptr;
  size_t code_cap_ = 0;
  size_t code_pos_ = 0;
  size_t code_reset_pos_ = 0;  // right after trampoline/epilogue

  EnterFn enter_ = nullptr;
  void* epilogue_ = nullptr;

  // Direct-mapped table indexed by (pc >> 2), 65536 entries (512 KB).
  BlockFn* blocks_ = nullptr;
  // Bit p (0..63) is set if any active block in blocks_ was compiled from 4KB LS page p.
  uint64_t code_pages_ = 0;

  // Content-addressed cache of compiled blocks: survives SPURS job DMA swaps so
  // previously compiled SPU jobs are re-bound in O(1) without x86-64 re-emission.
  struct BlockKey {
    uint32_t pc, count;
    uint64_t h1, h2;
    bool operator==(const BlockKey& o) const { return pc == o.pc && count == o.count && h1 == o.h1 && h2 == o.h2; }
  };
  struct BlockKeyHash {
    size_t operator()(const BlockKey& k) const noexcept {
      return size_t(k.h1 ^ (k.h2 * 0x9E3779B97F4A7C15ull) ^ (uint64_t(k.pc) << 32) ^ k.count);
    }
  };
  std::unordered_map<BlockKey, BlockFn, BlockKeyHash> block_cache_;
};

}  // namespace spu
