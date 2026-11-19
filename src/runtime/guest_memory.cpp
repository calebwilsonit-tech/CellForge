#include "guest_memory.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ppu {

namespace {
constexpr uint64_t kSpace = 1ull << 32;
constexpr uint64_t kPage = 4096;
}  // namespace

#if defined(_WIN32)

GuestMemory::GuestMemory() {
  base_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, kSpace, MEM_RESERVE, PAGE_NOACCESS));
}

GuestMemory::~GuestMemory() {
  if (base_) VirtualFree(base_, 0, MEM_RELEASE);
}

bool GuestMemory::map(uint32_t addr, uint64_t size) {
  if (!base_ || size == 0 || addr + size > kSpace) return false;
  const uint64_t start = addr & ~(kPage - 1);
  const uint64_t end = (addr + size + kPage - 1) & ~(kPage - 1);
  return VirtualAlloc(base_ + start, size_t(end - start), MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

#else

GuestMemory::GuestMemory() {
  void* p = mmap(nullptr, kSpace, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  base_ = (p == MAP_FAILED) ? nullptr : static_cast<uint8_t*>(p);
}

GuestMemory::~GuestMemory() {
  if (base_) munmap(base_, kSpace);
}

bool GuestMemory::map(uint32_t addr, uint64_t size) {
  if (!base_ || size == 0 || addr + size > kSpace) return false;
  const uint64_t start = addr & ~(kPage - 1);
  const uint64_t end = (addr + size + kPage - 1) & ~(kPage - 1);
  return mprotect(base_ + start, size_t(end - start), PROT_READ | PROT_WRITE) == 0;
}

#endif

}  // namespace ppu
