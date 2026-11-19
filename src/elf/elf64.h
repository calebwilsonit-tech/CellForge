#pragma once
// Minimal loader for big-endian ELF64 PowerPC executables (the plain, already
// decrypted form of a PS3 PPU executable). SELF/encrypted containers are
// rejected: this tool does not decrypt anything.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace elf {

struct Segment {
  uint64_t vaddr = 0;
  uint64_t memsz = 0;
  uint32_t flags = 0;          // PF_X = 1, PF_W = 2, PF_R = 4
  std::vector<uint8_t> data;   // always memsz bytes (zero padded past filesz)

  bool executable() const { return (flags & 1u) != 0; }
  bool contains(uint64_t addr, uint64_t len = 1) const {
    return addr >= vaddr && len <= memsz && addr - vaddr <= memsz - len;
  }
};

// Program header types used on PS3.
constexpr uint32_t PT_LOAD = 1;
constexpr uint32_t PT_TLS = 7;
constexpr uint32_t PT_PROC_PARAM = 0x60000001;  // sys_process_param_t
constexpr uint32_t PT_PRX_PARAM = 0x60000002;   // sys_process_prx_param (import/export tables)

struct ProgramHeader {
  uint32_t type = 0;
  uint32_t flags = 0;
  uint64_t vaddr = 0;
  uint64_t filesz = 0;
  uint64_t memsz = 0;
};

struct Image {
  uint64_t entry = 0;  // raw e_entry; on PS3 this points at an OPD, not code
  std::vector<Segment> segments;
  std::vector<ProgramHeader> phdrs;  // all program headers, including PT_LOAD

  const ProgramHeader* find_phdr(uint32_t type) const {
    for (const auto& h : phdrs)
      if (h.type == type) return &h;
    return nullptr;
  }
  bool read_cstr(uint64_t addr, std::string& out, size_t max = 256) const {
    out.clear();
    for (size_t i = 0; i < max; ++i) {
      const Segment* s = find(addr + i);
      if (!s) return false;
      const char ch = char(s->data[addr + i - s->vaddr]);
      if (!ch) return true;
      out += ch;
    }
    return true;
  }

  const Segment* find(uint64_t addr, uint64_t len = 1) const;
  bool is_executable(uint64_t addr) const;

  bool read16(uint64_t addr, uint16_t& out) const;
  bool read32(uint64_t addr, uint32_t& out) const;
  bool read64(uint64_t addr, uint64_t& out) const;

  // PS3 function descriptor (OPD): { u32 code address, u32 TOC pointer }.
  bool read_opd(uint64_t addr, uint64_t& code, uint64_t& toc) const;
};

struct LoadResult {
  bool ok = false;
  std::string error;
  Image image;
};

LoadResult load_memory(const uint8_t* data, size_t size);
LoadResult load_file(const std::string& path);

}  // namespace elf
