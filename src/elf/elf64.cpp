#include "elf64.h"

#include <cstring>
#include <fstream>

namespace elf {
namespace {

constexpr uint16_t kEmPpc64 = 21;
constexpr uint32_t kPtLoad = 1;
constexpr uint64_t kMaxSegment = 512ull << 20;   // sanity caps against malformed input
constexpr uint64_t kMaxTotal = 1ull << 30;

struct Reader {
  const uint8_t* p;
  size_t n;

  bool has(uint64_t off, uint64_t len) const { return off <= n && len <= n - off; }
  uint16_t u16(uint64_t o) const { return uint16_t((p[o] << 8) | p[o + 1]); }
  uint32_t u32(uint64_t o) const {
    return (uint32_t(p[o]) << 24) | (uint32_t(p[o + 1]) << 16) | (uint32_t(p[o + 2]) << 8) | p[o + 3];
  }
  uint64_t u64(uint64_t o) const { return (uint64_t(u32(o)) << 32) | u32(o + 4); }
};

LoadResult fail(std::string msg) {
  LoadResult r;
  r.error = std::move(msg);
  return r;
}

}  // namespace

const Segment* Image::find(uint64_t addr, uint64_t len) const {
  for (const auto& s : segments)
    if (s.contains(addr, len)) return &s;
  return nullptr;
}

bool Image::is_executable(uint64_t addr) const {
  const Segment* s = find(addr, 4);
  return s && s->executable();
}

bool Image::read16(uint64_t addr, uint16_t& out) const {
  const Segment* s = find(addr, 2);
  if (!s) return false;
  const uint8_t* b = &s->data[addr - s->vaddr];
  out = uint16_t((b[0] << 8) | b[1]);
  return true;
}

bool Image::read32(uint64_t addr, uint32_t& out) const {
  const Segment* s = find(addr, 4);
  if (!s) return false;
  const uint8_t* b = &s->data[addr - s->vaddr];
  out = (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
  return true;
}

bool Image::read64(uint64_t addr, uint64_t& out) const {
  uint32_t hi = 0, lo = 0;
  if (!read32(addr, hi) || !read32(addr + 4, lo)) return false;
  out = (uint64_t(hi) << 32) | lo;
  return true;
}

bool Image::read_opd(uint64_t addr, uint64_t& code, uint64_t& toc) const {
  uint32_t c = 0, t = 0;
  if (!read32(addr, c) || !read32(addr + 4, t)) return false;
  code = c;
  toc = t;
  return true;
}

LoadResult load_memory(const uint8_t* data, size_t size) {
  if (size >= 4 && std::memcmp(data, "SCE\0", 4) == 0)
    return fail("input is a SELF (signed/encrypted PS3 executable), not a plain ELF. "
                "This tool only reads already-decrypted ELF files and does not decrypt anything.");
  if (size < 64 || std::memcmp(data, "\x7f" "ELF", 4) != 0) return fail("not an ELF file");

  Reader r{data, size};
  if (data[4] != 2) return fail("not a 64-bit ELF");
  if (data[5] != 2) return fail("not a big-endian ELF");
  if (r.u16(18) != kEmPpc64) return fail("not a PowerPC64 ELF (e_machine != 21)");

  LoadResult out;
  out.image.entry = r.u64(24);
  const uint64_t phoff = r.u64(32);
  const uint16_t phentsize = r.u16(54);
  const uint16_t phnum = r.u16(56);
  if (phentsize < 56) return fail("unsupported program header size");
  if (!r.has(phoff, uint64_t(phentsize) * phnum)) return fail("program headers out of bounds");

  uint64_t total = 0;
  for (uint16_t i = 0; i < phnum; ++i) {
    const uint64_t ph = phoff + uint64_t(i) * phentsize;
    ProgramHeader h;
    h.type = r.u32(ph);
    h.flags = r.u32(ph + 4);
    h.vaddr = r.u64(ph + 16);
    h.filesz = r.u64(ph + 32);
    h.memsz = r.u64(ph + 40);
    out.image.phdrs.push_back(h);
    if (h.type != kPtLoad || h.memsz == 0) continue;  // PS3 toolchains emit empty PT_LOADs at 0

    Segment s;
    s.flags = r.u32(ph + 4);
    const uint64_t offset = r.u64(ph + 8);
    s.vaddr = r.u64(ph + 16);
    const uint64_t filesz = r.u64(ph + 32);
    s.memsz = r.u64(ph + 40);

    if (filesz > s.memsz) return fail("segment filesz > memsz");
    if (s.memsz > kMaxSegment || (total += s.memsz) > kMaxTotal) return fail("segment too large");
    if (s.vaddr > 0xFFFFFFFFull || s.memsz > 0x100000000ull - s.vaddr)
      return fail("segment outside the 32-bit PS3 address space");
    if (!r.has(offset, filesz)) return fail("segment data out of bounds");

    s.data.assign(size_t(s.memsz), 0);
    if (filesz) std::memcpy(s.data.data(), data + offset, size_t(filesz));
    out.image.segments.push_back(std::move(s));
  }
  if (out.image.segments.empty()) return fail("no PT_LOAD segments");
  out.ok = true;
  return out;
}

LoadResult load_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return fail("cannot open " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return load_memory(buf.data(), buf.size());
}

}  // namespace elf
