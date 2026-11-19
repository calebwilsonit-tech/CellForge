#include "nid.h"

#include <cstring>
#include <unordered_map>
#include <vector>

namespace nid {
namespace {

uint32_t rol(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }

const char* const kNames[] = {
#define N(x) #x,
#include "nid_names.inc"
#undef N
};

const std::unordered_map<uint32_t, std::string>& db() {
  static const std::unordered_map<uint32_t, std::string> m = [] {
    std::unordered_map<uint32_t, std::string> r;
    for (const char* n : kNames) r.emplace(of(n), n);
    return r;
  }();
  return m;
}

}  // namespace

std::array<uint8_t, 20> sha1(const void* data, size_t len) {
  uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  std::vector<uint8_t> msg(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + len);
  const uint64_t bits = uint64_t(len) * 8;
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back(uint8_t(bits >> (8 * i)));

  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(msg[off + 4 * i]) << 24) | (uint32_t(msg[off + 4 * i + 1]) << 16) |
             (uint32_t(msg[off + 4 * i + 2]) << 8) | msg[off + 4 * i + 3];
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else { f = b ^ c ^ d; k = 0xCA62C1D6; }
      const uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
  }
  std::array<uint8_t, 20> out{};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 4; ++j) out[4 * i + j] = uint8_t(h[i] >> (24 - 8 * j));
  return out;
}

uint32_t of(std::string_view name) {
  static const uint8_t kSuffix[16] = {0x67, 0x59, 0x65, 0x99, 0x04, 0x25, 0x04, 0x90,
                                      0x56, 0x64, 0x27, 0x49, 0x94, 0x89, 0x74, 0x1A};
  std::vector<uint8_t> buf(name.begin(), name.end());
  buf.insert(buf.end(), kSuffix, kSuffix + 16);
  const auto d = sha1(buf.data(), buf.size());
  return uint32_t(d[0]) | (uint32_t(d[1]) << 8) | (uint32_t(d[2]) << 16) | (uint32_t(d[3]) << 24);
}

const std::string& name_of(uint32_t n) {
  static const std::string empty;
  const auto& m = db();
  auto it = m.find(n);
  return it == m.end() ? empty : it->second;
}

}  // namespace nid
