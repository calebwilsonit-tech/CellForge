#pragma once
#include <cstddef>
#include <cstdint>

namespace ppu {

// Reserves a 4 GiB address space and commits ranges on demand, so that a guest
// 32-bit address `a` is simply `base() + a`.
class GuestMemory {
 public:
  GuestMemory();
  ~GuestMemory();
  GuestMemory(const GuestMemory&) = delete;
  GuestMemory& operator=(const GuestMemory&) = delete;

  uint8_t* base() const { return base_; }
  bool ok() const { return base_ != nullptr; }

  // Commit [addr, addr+size) as read/write. Safe to call on overlapping ranges.
  bool map(uint32_t addr, uint64_t size);

 private:
  uint8_t* base_ = nullptr;
};

}  // namespace ppu
