#pragma once
// x86-64 AVX2 basic-block JIT for the Cell SPU.
// Active on Windows x64; other platforms fall back to the interpreter.

#include <cstdint>

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
  // Bit p (0..63) is set if any block was compiled from 4KB LS page p.
  uint64_t code_pages_ = 0;
};

}  // namespace spu
