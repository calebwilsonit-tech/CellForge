#pragma once
// SPU (Synergistic Processing Unit) interpreter, written from the public Cell
// Broadband Engine SPU ISA. One SpuCore = one SPU: 256 KB local store, 128
// 128-bit registers, and a channel interface that the host (thread-group / MFC
// layer) implements.

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>

namespace spu {

inline uint32_t bswap_u32(uint32_t v) {
#if defined(_MSC_VER)
  return _byteswap_ulong(v);
#else
  return __builtin_bswap32(v);
#endif
}

constexpr uint32_t kLsSize = 0x40000;
constexpr uint32_t kLsMask = kLsSize - 1;

// A 128-bit register held as four native 32-bit words, word 0 = the preferred
// slot (the most significant word in guest byte order). Element-wise operations
// work directly on this layout (and map onto SSE lanes); byte-order-sensitive
// operations go through byte()/set_byte(), where byte 0 is the guest's
// most significant byte.
struct V128 {
  alignas(16) uint32_t u[4];

  uint32_t w(int i) const { return u[i]; }
  void set_w(int i, uint32_t v) { u[i] = v; }
  uint16_t h(int i) const { return uint16_t((i & 1) ? u[i >> 1] : u[i >> 1] >> 16); }
  void set_h(int i, uint16_t v) {
    uint32_t& x = u[i >> 1];
    x = (i & 1) ? (x & 0xFFFF0000u) | v : (x & 0xFFFFu) | (uint32_t(v) << 16);
  }
  uint8_t byte(int i) const { return uint8_t(u[i >> 2] >> (24 - 8 * (i & 3))); }
  void set_byte(int i, uint8_t v) {
    const int s = 24 - 8 * (i & 3);
    u[i >> 2] = (u[i >> 2] & ~(0xFFu << s)) | (uint32_t(v) << s);
  }
  uint64_t d(int i) const { return uint64_t(u[2 * i]) << 32 | u[2 * i + 1]; }
  void set_d(int i, uint64_t v) { u[2 * i] = uint32_t(v >> 32); u[2 * i + 1] = uint32_t(v); }
  float f(int i) const { return std::bit_cast<float>(u[i]); }
  void set_f(int i, float x) { u[i] = std::bit_cast<uint32_t>(x); }
  double df(int i) const { return std::bit_cast<double>(d(i)); }
  void set_df(int i, double x) { set_d(i, std::bit_cast<uint64_t>(x)); }

  static V128 splat_w(uint32_t v) { V128 r; r.u[0] = r.u[1] = r.u[2] = r.u[3] = v; return r; }
  static V128 zero() { return splat_w(0); }
};

// Channel numbers (CBEA).
enum Channel : uint32_t {
  SPU_RdEventStat = 0, SPU_WrEventMask = 1, SPU_WrEventAck = 2,
  SPU_RdSigNotify1 = 3, SPU_RdSigNotify2 = 4,
  SPU_WrDec = 7, SPU_RdDec = 8,
  MFC_WrMSSyncReq = 9, SPU_RdEventMask = 11, MFC_RdTagMask = 12, SPU_RdMachStat = 13,
  SPU_WrSRR0 = 14, SPU_RdSRR0 = 15,
  MFC_LSA = 16, MFC_EAH = 17, MFC_EAL = 18, MFC_Size = 19, MFC_TagID = 20, MFC_Cmd = 21,
  MFC_WrTagMask = 22, MFC_WrTagUpdate = 23, MFC_RdTagStat = 24, MFC_RdListStallStat = 25,
  MFC_WrListStallAck = 26, MFC_RdAtomicStat = 27,
  SPU_WrOutMbox = 28, SPU_RdInMbox = 29, SPU_WrOutIntrMbox = 30,
};

class SpuCore;
class SpuJit;

// What an SPU needs from the outside world. Reads may block (mailboxes, events).
struct SpuHost {
  virtual ~SpuHost() = default;
  virtual uint32_t read_channel(SpuCore& s, uint32_t ch) = 0;
  virtual void write_channel(SpuCore& s, uint32_t ch, uint32_t v) = 0;
  virtual uint32_t channel_count(SpuCore& s, uint32_t ch) = 0;
  // `stop` / `stopd` with a 14-bit code. Return true to continue at pc+4.
  virtual bool stop(SpuCore& s, uint32_t code) = 0;
};

struct SpuHalt {
  uint32_t pc;
  const char* why;
};

class SpuCore {
 public:
  explicit SpuCore(SpuHost* host);
  ~SpuCore();

  uint8_t* ls() { return ls_.get(); }
  V128 gpr[128];
  uint32_t pc = 0;
  uint32_t srr0 = 0;
  uint32_t dec = 0;
  int64_t dec_base_ns = 0;
  uint32_t jit_stop_code_ = 0;
  std::exception_ptr jit_exception_;
  SpuHost* host;

  uint32_t ls32(uint32_t a) const {
    uint32_t v;
    std::memcpy(&v, ls_.get() + (a & kLsMask & ~3u), 4);
    return bswap_u32(v);
  }
  void set_ls32(uint32_t a, uint32_t v) {
    v = bswap_u32(v);
    std::memcpy(ls_.get() + (a & kLsMask & ~3u), &v, 4);
    invalidate_ls(a & kLsMask & ~3u, 4);
  }

  // Executes until a `stop` the host does not continue from. Returns its code.
  // Throws SpuHalt on halt instructions / invalid opcodes.
  uint32_t run();
  // Executes one instruction; returns false (and sets *stop_code) on a final stop.
  bool step(uint32_t* stop_code);

  // Notify JIT that Local Store [addr, addr+len) was modified (e.g. by DMA GET).
  void invalidate_ls(uint32_t addr, uint32_t len);
  void invalidate_all();

  // Clears registers and state for a fresh run (the local store is left as is).
  void reset() {
    for (auto& r : gpr) r = V128::zero();
    pc = srr0 = dec = 0;
    exit_requested_ = false;
    invalidate_all();
  }
  void request_exit() { exit_requested_ = true; }
  bool exit_requested() const { return exit_requested_; }

 private:
  std::unique_ptr<uint8_t[]> ls_;
  std::unique_ptr<SpuJit> jit_;
  volatile bool exit_requested_ = false;
};

}  // namespace spu
