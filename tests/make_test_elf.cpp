// Hand-assembles a tiny big-endian PPC64 ELF used by the end-to-end test.
// Writes <outdir>/test.elf, <outdir>/test_funcs.txt ("<name> <hex addr>", the
// recompiler's seeds) and <outdir>/test_layout.h (addresses for the test + mod).
//
// No game data is involved: every instruction below is written by hand.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "nid.h"

namespace {

constexpr uint32_t kBase = 0x10000;
constexpr uint32_t kPhnum = 4;
constexpr uint32_t kOpdOff = 0x140;                 // after the ELF + program headers
constexpr uint32_t kCodeOff = 0x200;
constexpr uint32_t kDataOff = 0x1000;
constexpr uint32_t kDataSize = 0x400;
constexpr uint32_t kToc = kBase + kDataOff;         // r2
constexpr uint32_t kDataOpd = kDataOff + 0x20;      // OPD for a function reachable only by pointer
constexpr uint32_t kWorkerOpd = kDataOff + 0x30;    // OPD of the worker thread entry
constexpr uint32_t kAbsTable = kDataOff + 0x40;
constexpr uint32_t kRelTable = kDataOff + 0x80;
constexpr uint32_t kProcParam = kDataOff + 0x100;   // sys_process_param_t
constexpr uint32_t kPrxParam = kDataOff + 0x140;    // sys_process_prx_param
constexpr uint32_t kLibStub = kDataOff + 0x180;     // 2 x 0x2C import module entries
constexpr uint32_t kModNames = kDataOff + 0x1E0;
constexpr uint32_t kNids = kDataOff + 0x200;
constexpr uint32_t kFooNids = kDataOff + 0x240;
constexpr uint32_t kSlots = kDataOff + 0x260;
constexpr uint32_t kFooSlots = kDataOff + 0x280;
constexpr uint32_t kTlsImage = kDataOff + 0x2A0;
constexpr uint32_t kFmt = kDataOff + 0x2C0;
constexpr uint32_t kGuestStr = kDataOff + 0x2D0;
constexpr uint32_t kTtyMsg = kDataOff + 0x2E0;
constexpr uint32_t kCounter = kDataOff + 0x300;
constexpr uint32_t kResults = kDataOff + 0x320;

const char* const kSysImports[] = {"sys_initialize_tls", "_sys_printf", "sys_time_get_system_time",
                                   "sys_ppu_thread_create"};
constexpr uint32_t kFooNid = 0xDEADBEEF;  // an import nobody implements

// ---- encoders -------------------------------------------------------------
uint32_t dform(int op, int rt, int ra, int imm) { return (uint32_t(op) << 26) | (rt << 21) | (ra << 16) | (imm & 0xFFFF); }
uint32_t xform(int op, int rt, int ra, int rb, int xo) {
  return (uint32_t(op) << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1);
}
uint32_t addi(int rt, int ra, int s) { return dform(14, rt, ra, s); }
uint32_t li(int rt, int s) { return addi(rt, 0, s); }
uint32_t lis(int rt, int s) { return dform(15, rt, 0, s); }
uint32_t ori(int ra, int rs, int u) { return dform(24, rs, ra, u); }
uint32_t cmpwi(int bf, int ra, int s) { return (11u << 26) | (bf << 23) | (ra << 16) | (s & 0xFFFF); }
uint32_t cmplwi(int bf, int ra, int u) { return (10u << 26) | (bf << 23) | (ra << 16) | (u & 0xFFFF); }
uint32_t bc(int bo, int bi, int delta) { return (16u << 26) | (bo << 21) | (bi << 16) | (delta & 0xFFFC); }
uint32_t b(int delta) { return (18u << 26) | (delta & 0x03FFFFFC); }
uint32_t bl(int delta) { return b(delta) | 1u; }
uint32_t add(int rt, int ra, int rb) { return xform(31, rt, ra, rb, 266); }
uint32_t addc(int rt, int ra, int rb) { return xform(31, rt, ra, rb, 10); }
uint32_t addze(int rt, int ra) { return xform(31, rt, ra, 0, 202); }
uint32_t mr(int ra, int rs) { return xform(31, rs, ra, rs, 444); }
uint32_t mflr(int rt) { return (31u << 26) | (rt << 21) | (256u << 11) | (339u << 1); }
uint32_t mtlr(int rs) { return (31u << 26) | (rs << 21) | (256u << 11) | (467u << 1); }
uint32_t mtctr(int rs) { return (31u << 26) | (rs << 21) | (288u << 11) | (467u << 1); }
uint32_t mfcr(int rt) { return xform(31, rt, 0, 0, 19); }
uint32_t blr() { return 0x4E800020u; }
uint32_t bctr() { return 0x4E800420u; }
uint32_t bctrl() { return 0x4E800421u; }
uint32_t rlwinm(int ra, int rs, int sh, int mb, int me) {
  return (21u << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1);
}
uint32_t rldicr(int ra, int rs, int sh, int me) {
  return (30u << 26) | (rs << 21) | (ra << 16) | ((sh & 31) << 11) | ((me & 31) << 6) | ((me >> 5) << 5) | (1 << 2) |
         ((sh >> 5) << 1);
}
uint32_t srawi(int ra, int rs, int sh) { return (31u << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (824u << 1); }
uint32_t stw(int rs, int ra, int d) { return dform(36, rs, ra, d); }
uint32_t lwz(int rt, int ra, int d) { return dform(32, rt, ra, d); }
uint32_t lbz(int rt, int ra, int d) { return dform(34, rt, ra, d); }
uint32_t ld(int rt, int ra, int ds) { return (58u << 26) | (rt << 21) | (ra << 16) | (ds & 0xFFFC); }
uint32_t std_(int rs, int ra, int ds) { return (62u << 26) | (rs << 21) | (ra << 16) | (ds & 0xFFFC); }
uint32_t stdu(int rs, int ra, int ds) { return std_(rs, ra, ds) | 1u; }
uint32_t lwzx(int rt, int ra, int rb) { return xform(31, rt, ra, rb, 23); }
uint32_t lwax(int rt, int ra, int rb) { return xform(31, rt, ra, rb, 341); }
uint32_t lwarx(int rt, int ra, int rb) { return xform(31, rt, ra, rb, 20); }
uint32_t stwcx(int rs, int ra, int rb) { return xform(31, rs, ra, rb, 150) | 1u; }
// floating point
uint32_t lfd(int ft, int ra, int d) { return dform(50, ft, ra, d); }
uint32_t stfd(int fs, int ra, int d) { return dform(54, fs, ra, d); }
uint32_t fmadd(int ft, int fa, int fc, int fb) {
  return (63u << 26) | (ft << 21) | (fa << 16) | (fb << 11) | (fc << 6) | (29u << 1);
}
uint32_t fctiwz(int ft, int fb) { return (63u << 26) | (ft << 21) | (fb << 11) | (15u << 1); }
uint32_t fcmpu(int bf, int fa, int fb) { return (63u << 26) | (bf << 23) | (fa << 16) | (fb << 11); }
// vector
uint32_t lvx(int vd, int ra, int rb) { return xform(31, vd, ra, rb, 103); }
uint32_t stvx(int vs, int ra, int rb) { return xform(31, vs, ra, rb, 231); }
uint32_t lvsl(int vd, int ra, int rb) { return xform(31, vd, ra, rb, 6); }
uint32_t vx(int vd, int va, int vb, int xo) { return (4u << 26) | (vd << 21) | (va << 16) | (vb << 11) | xo; }
uint32_t vaddfp(int vd, int va, int vb) { return vx(vd, va, vb, 10); }
uint32_t vspltw(int vd, int vb, int u) { return vx(vd, u, vb, 652); }
uint32_t vperm(int vd, int va, int vb, int vc) { return vx(vd, va, vb, (vc << 6) | 43); }
uint32_t vcmpgtfp_rc(int vd, int va, int vb) { return vx(vd, va, vb, (1 << 10) | 710); }
uint32_t sc() { return 0x44000002u; }
int ha(uint32_t a) { return int((a + 0x8000) >> 16); }
int lo(uint32_t a) { return int(int16_t(a & 0xFFFF)); }

// ---- image builder --------------------------------------------------------
struct Func {
  std::string name;
  uint32_t addr;
  bool seed;
};

std::vector<uint32_t> code;
std::vector<Func> funcs;

uint32_t here() { return kBase + kCodeOff + uint32_t(code.size()) * 4; }
void begin(const std::string& name, bool seed = true) { funcs.push_back({name, here(), seed}); }
void emit(uint32_t insn) { code.push_back(insn); }
uint32_t addr_of(const std::string& name) {
  for (const auto& f : funcs)
    if (f.name == name) return f.addr;
  std::fprintf(stderr, "unknown function %s\n", name.c_str());
  std::exit(1);
}
void put32(std::vector<uint8_t>& v, size_t off, uint32_t x) {
  v[off] = uint8_t(x >> 24); v[off + 1] = uint8_t(x >> 16); v[off + 2] = uint8_t(x >> 8); v[off + 3] = uint8_t(x);
}
void put16(std::vector<uint8_t>& v, size_t off, uint16_t x) { v[off] = uint8_t(x >> 8); v[off + 1] = uint8_t(x); }
void put64(std::vector<uint8_t>& v, size_t off, uint64_t x) { put32(v, off, uint32_t(x >> 32)); put32(v, off + 4, uint32_t(x)); }

void la(int r, uint32_t addr) { emit(lis(r, ha(addr))); emit(addi(r, r, lo(addr))); }
// The standard PS3 import call stub: load the OPD pointer from the import slot,
// save r2, call through the OPD.
void import_stub(const std::string& name, uint32_t slot) {
  begin(name, false);
  emit(lis(12, ha(slot)));
  emit(lwz(12, 12, lo(slot)));
  emit(std_(2, 1, 40));
  emit(lwz(0, 12, 0));
  emit(lwz(2, 12, 4));
  emit(mtctr(0));
  emit(bctr());
}
void call(const std::string& fn) { emit(bl(int(addr_of(fn)) - int(here()))); emit(ld(2, 1, 40)); }
// 1000 atomic increments of the word at (rp), using ctr.
void atomic_add_1000(int rp) {
  emit(li(5, 1000));
  emit(mtctr(5));
  emit(lwarx(4, 0, rp));
  emit(addi(4, 4, 1));
  emit(stwcx(4, 0, rp));
  emit(bc(4, 2, -12));    // bne- retry
  emit(bc(16, 0, -16));   // bdnz loop
}

void prologue() { emit(mflr(0)); emit(stdu(1, 1, -48)); emit(std_(0, 1, 64)); }
void epilogue() { emit(ld(0, 1, 64)); emit(addi(1, 1, 48)); emit(mtlr(0)); emit(blr()); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: make_test_elf <outdir>\n"); return 2; }
  const std::string dir = argv[1];

  // sum(n): loop with compare + conditional branch
  begin("sum");
  emit(li(4, 0));
  emit(cmpwi(0, 3, 0));         // loop:
  emit(bc(12, 2, 16));          // beq done
  emit(add(4, 4, 3));
  emit(addi(3, 3, -1));
  emit(b(-16));                 // b loop
  emit(mr(3, 4));               // done:
  emit(blr());

  // call_sum(): stack frame + bl. Returns sum(10) = 55.
  begin("call_sum");
  prologue();
  emit(li(3, 10));
  emit(bl(int(addr_of("sum")) - int(here())));
  epilogue();

  // call_indirect(): mtctr + bctrl. Returns sum(7) = 28.
  begin("call_indirect");
  prologue();
  emit(li(3, 7));
  emit(lis(12, int(addr_of("sum") >> 16)));
  emit(ori(12, 12, int(addr_of("sum") & 0xFFFF)));
  emit(mtctr(12));
  emit(bctrl());
  epilogue();

  // shl2(x): slwi r3,r3,2 via rlwinm
  begin("shl2");
  emit(rlwinm(3, 3, 2, 0, 29));
  emit(blr());

  // store_load(x, addr): big-endian memory
  begin("store_load");
  emit(stw(3, 4, 0));
  emit(lbz(3, 4, 0));
  emit(blr());

  // sw_abs(x): switch through an absolute jump table
  begin("sw_abs");
  {
    const uint32_t f = here();
    emit(cmplwi(7, 3, 3));
    emit(bc(12, 29, 15 * 4));                 // bgt cr7, default (index 16)
    emit(lis(9, int((kBase + kAbsTable) >> 16)));
    emit(ori(9, 9, int((kBase + kAbsTable) & 0xFFFF)));
    emit(rlwinm(10, 3, 2, 0, 29));
    emit(lwzx(11, 9, 10));
    emit(mtctr(11));
    emit(bctr());
    for (int k = 0; k < 4; ++k) { emit(li(3, 100 + k)); emit(blr()); }
    emit(li(3, -1));                          // default
    emit(blr());
    (void)f;
  }

  // sw_rel(x): switch through a TOC-loaded, relative (lwax + add) table
  begin("sw_rel");
  emit(cmplwi(0, 3, 2));
  emit(bc(12, 1, 13 * 4));                    // bgt cr0, default (index 14)
  emit(lwz(9, 2, 0));                         // r9 = table address from the TOC
  emit(rldicr(10, 3, 2, 61));                 // sldi r10,r3,2
  emit(lwax(11, 9, 10));
  emit(add(11, 11, 9));
  emit(mtctr(11));
  emit(bctr());
  for (int k = 0; k < 3; ++k) { emit(li(3, 200 + k)); emit(blr()); }
  emit(li(3, -1));
  emit(blr());

  // fp(p): fmadd, fctiwz, stfd, fcmpu, mfcr
  begin("fp");
  emit(lfd(1, 3, 0));
  emit(lfd(2, 3, 8));
  emit(fmadd(3, 1, 2, 1));                    // f3 = f1*f2 + f1
  emit(fctiwz(4, 3));
  emit(stfd(4, 3, 16));
  emit(lwz(4, 3, 20));                        // low word of the integer result
  emit(fcmpu(1, 1, 2));
  emit(mfcr(5));
  emit(rlwinm(5, 5, 5, 31, 31));              // cr1.lt
  emit(mr(3, 4));
  emit(mr(4, 5));
  emit(blr());

  // carry(a, b): addc + addze -> r3 = carry, r4 = sum
  begin("carry");
  emit(addc(5, 3, 4));
  emit(li(6, 0));
  emit(addze(3, 6));
  emit(mr(4, 5));
  emit(blr());

  // sra(x): srawi + CA
  begin("sra");
  emit(srawi(3, 3, 4));
  emit(li(4, 0));
  emit(addze(4, 4));
  emit(blr());

  // vmx(p): lvx, vaddfp, vspltw, lvsl+vperm (unaligned load), vcmpgtfp.
  begin("vmx");
  emit(li(4, 16));
  emit(lvx(1, 0, 3));
  emit(lvx(2, 3, 4));
  emit(vaddfp(3, 1, 2));
  emit(li(5, 32));
  emit(stvx(3, 3, 5));
  emit(vspltw(4, 1, 2));
  emit(li(5, 48));
  emit(stvx(4, 3, 5));
  emit(addi(6, 3, 1));
  emit(lvsl(5, 0, 6));
  emit(vperm(6, 1, 2, 5));
  emit(li(5, 64));
  emit(stvx(6, 3, 5));
  emit(vcmpgtfp_rc(7, 2, 1));
  emit(mfcr(3));
  emit(rlwinm(3, 3, 25, 31, 31));             // cr6.lt: all elements true
  emit(blr());

  // atomic_inc(p): lwarx / stwcx. retry loop
  begin("atomic_inc");
  emit(lwarx(4, 0, 3));                       // loop:
  emit(addi(4, 4, 1));
  emit(stwcx(4, 0, 3));
  emit(bc(4, 2, -12));                        // bne loop
  emit(mr(3, 4));
  emit(blr());

  // opd_only(): only reachable through an OPD in the data area (tests OPD scanning)
  begin("opd_only", false);
  emit(li(3, 77));
  emit(blr());

  // ---- OS-level test: imports, TLS, printf, threads, syscalls ----
  for (int i = 0; i < 4; ++i) import_stub(std::string("stub_") + kSysImports[i], kBase + kSlots + 4 * i);
  import_stub("stub_foo", kBase + kFooSlots);

  // worker(counter): 1000 atomic increments, then returns 7 (the thread's exit value)
  begin("worker", false);
  atomic_add_1000(3);
  emit(li(3, 7));
  emit(blr());

  // start(): the process entry point (via the ELF entry OPD)
  begin("start", false);
  emit(mflr(0));
  emit(stdu(1, 1, -128));
  emit(std_(0, 1, 144));
  emit(mr(3, 7)); emit(mr(4, 8)); emit(mr(5, 9)); emit(mr(6, 10));  // loader passes TLS info in r7..r10
  call("stub_sys_initialize_tls");
  la(20, kBase + kDataOff);
  emit(lwz(14, 13, -0x7000));                 // first TLS word
  emit(stw(14, 20, kResults - kDataOff + 0));
  la(3, kBase + kFmt);
  la(4, kBase + kGuestStr);
  emit(li(5, 42));
  call("stub__sys_printf");
  emit(stw(3, 20, kResults - kDataOff + 20));  // printf return value (length)
  call("stub_sys_time_get_system_time");
  emit(stw(3, 20, kResults - kDataOff + 4));
  emit(li(3, 5));
  call("stub_foo");
  emit(stw(3, 20, kResults - kDataOff + 8));
  // sys_ppu_thread_create(&tid, worker_opd, counter, prio, stack, JOINABLE, name)
  emit(addi(3, 1, 112));
  la(4, kBase + kWorkerOpd);
  la(5, kBase + kCounter);
  emit(li(6, 1000));
  emit(lis(7, 1));
  emit(li(8, 1));
  emit(li(9, 0));
  call("stub_sys_ppu_thread_create");
  la(16, kBase + kCounter);
  atomic_add_1000(16);                        // race the worker
  emit(ld(3, 1, 112));                        // sys_ppu_thread_join(tid, &exit)
  emit(addi(4, 1, 120));
  emit(li(11, 44));
  emit(sc());
  emit(ld(5, 1, 120));
  emit(stw(5, 20, kResults - kDataOff + 12));
  emit(lwz(5, 20, kCounter - kDataOff));
  emit(stw(5, 20, kResults - kDataOff + 16));
  emit(li(3, 0));                             // sys_tty_write(0, msg, 7, &written)
  la(4, kBase + kTtyMsg);
  emit(li(5, 7));
  emit(addi(6, 1, 104));
  emit(li(11, 403));
  emit(sc());
  emit(li(3, 0));                             // sys_process_exit(0)
  emit(li(11, 22));
  emit(sc());
  emit(blr());

  if (kCodeOff + code.size() * 4 > kDataOff) { std::fprintf(stderr, "code overflows into data\n"); return 1; }

  // ---- assemble the ELF ----
  const uint32_t filesz = kDataOff + kDataSize;
  std::vector<uint8_t> img(filesz, 0);

  img[0] = 0x7F; img[1] = 'E'; img[2] = 'L'; img[3] = 'F';
  img[4] = 2; img[5] = 2; img[6] = 1;
  put16(img, 16, 2);    // ET_EXEC
  put16(img, 18, 21);   // EM_PPC64
  put32(img, 20, 1);
  put64(img, 24, kBase + kOpdOff);  // entry points at an OPD, like a real PS3 ELF
  put64(img, 32, 64);
  put16(img, 52, 64);
  put16(img, 54, 56);
  put16(img, 56, kPhnum);
  auto phdr = [&](int i, uint32_t type, uint32_t flags, uint32_t off, uint32_t fsz, uint32_t msz) {
    const size_t p = 64 + size_t(i) * 56;
    put32(img, p, type);
    put32(img, p + 4, flags);
    put64(img, p + 8, off);
    put64(img, p + 16, kBase + off);
    put64(img, p + 24, kBase + off);
    put64(img, p + 32, fsz);
    put64(img, p + 40, msz);
    put64(img, p + 48, type == 1 ? 0x1000 : 8);
  };
  phdr(0, 1, 5, 0, filesz, filesz);                  // PT_LOAD (R+X; data is written too)
  phdr(1, 0x60000001, 4, kProcParam, 0x24, 0x24);     // PT_PROC_PARAM
  phdr(2, 0x60000002, 4, kPrxParam, 0x40, 0x40);      // PT_PRX_PARAM
  phdr(3, 7, 4, kTlsImage, 8, 16);                    // PT_TLS

  // OPDs: { u32 code, u32 toc }
  put32(img, kOpdOff, addr_of("start"));
  put32(img, kOpdOff + 4, kToc);
  put32(img, kDataOpd, addr_of("opd_only"));
  put32(img, kDataOpd + 4, kToc);
  put32(img, kWorkerOpd, addr_of("worker"));
  put32(img, kWorkerOpd + 4, kToc);

  // sys_process_param_t
  put32(img, kProcParam + 0, 0x24);
  put32(img, kProcParam + 4, 0x13BCC5F6);
  put32(img, kProcParam + 8, 0x00009000);
  put32(img, kProcParam + 12, 0x00360001);  // SDK version
  put32(img, kProcParam + 16, 1001);        // primary thread priority
  put32(img, kProcParam + 20, 0x10000);     // primary stack size
  put32(img, kProcParam + 24, 0x100000);    // malloc page size
  // sys_process_prx_param
  put32(img, kPrxParam + 0, 0x40);
  put32(img, kPrxParam + 4, 0x1B434CEC);
  put32(img, kPrxParam + 8, 4);
  put32(img, kPrxParam + 24, kBase + kLibStub);
  put32(img, kPrxParam + 28, kBase + kLibStub + 2 * 0x2C);
  // import modules
  auto libstub = [&](int i, uint32_t name, uint16_t nfunc, uint32_t nids, uint32_t slots) {
    const size_t p = kLibStub + size_t(i) * 0x2C;
    img[p] = 0x2C;
    put16(img, p + 2, 1);
    put16(img, p + 4, 9);
    put16(img, p + 6, nfunc);
    put32(img, p + 0x10, kBase + name);
    put32(img, p + 0x14, kBase + nids);
    put32(img, p + 0x18, kBase + slots);
  };
  libstub(0, kModNames, 4, kNids, kSlots);
  libstub(1, kModNames + 0x10, 1, kFooNids, kFooSlots);
  std::memcpy(&img[kModNames], "sysPrxForUser", 13);
  std::memcpy(&img[kModNames + 0x10], "cellFoo", 7);
  for (int i = 0; i < 4; ++i) {
    put32(img, kNids + 4 * i, nid::of(kSysImports[i]));
    put32(img, kSlots + 4 * i, addr_of(std::string("stub_") + kSysImports[i]));
  }
  put32(img, kFooNids, kFooNid);
  put32(img, kFooSlots, addr_of("stub_foo"));
  // TLS image, strings
  put32(img, kTlsImage, 0x12345678);
  put32(img, kTlsImage + 4, 0x9ABCDEF0);
  std::memcpy(&img[kFmt], "hello %s %d\n", 12);
  std::memcpy(&img[kGuestStr], "guest", 5);
  std::memcpy(&img[kTtyMsg], "tty ok\n", 7);

  for (size_t i = 0; i < code.size(); ++i) put32(img, kCodeOff + i * 4, code[i]);

  // TOC[0] = relative table address
  put32(img, kDataOff, kBase + kRelTable);
  // absolute table: case k at sw_abs + 32 + 8k
  for (int k = 0; k < 4; ++k) put32(img, kAbsTable + 4 * k, addr_of("sw_abs") + 32 + 8 * k);
  // relative table: (case - table)
  for (int k = 0; k < 3; ++k)
    put32(img, kRelTable + 4 * k, uint32_t(int32_t(addr_of("sw_rel") + 32 + 8 * k) - int32_t(kBase + kRelTable)));

  std::ofstream(dir + "/test.elf", std::ios::binary)
      .write(reinterpret_cast<const char*>(img.data()), std::streamsize(img.size()));

  std::ofstream f(dir + "/test_funcs.txt");
  for (const auto& fn : funcs)
    if (fn.seed) f << fn.name << " 0x" << std::hex << fn.addr << "\n";

  std::ofstream h(dir + "/test_layout.h");
  h << "#pragma once\n// Generated by make_test_elf.\n#include <cstdint>\nnamespace test_layout {\n";
  h << "constexpr uint32_t kToc = 0x" << std::hex << kToc << ";\n";
  h << "constexpr uint32_t kResults = 0x" << std::hex << (kBase + kResults) << ";\n";
  h << "constexpr uint32_t kCounter = 0x" << std::hex << (kBase + kCounter) << ";\n";
  h << "constexpr uint32_t kFooNid = 0x" << std::hex << kFooNid << ";\n";
  for (const auto& fn : funcs) h << "constexpr uint32_t k_" << fn.name << " = 0x" << std::hex << fn.addr << ";\n";
  h << "}\n";
  return 0;
}
