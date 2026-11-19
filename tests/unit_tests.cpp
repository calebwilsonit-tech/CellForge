// Unit tests: decoder table consistency and VMX helper semantics.

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "ppu_decode.h"
#include "ppu_runtime.h"
#include "rsx_draw.h"
#include "rsx_program.h"
#include "spirv_gen.h"
#include "vk_renderer.h"

namespace {

int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) {
  ++g_checks;
  if (!ok) { std::printf("  FAIL  %s\n", what); ++g_fail; }
}

enum Form { PRIM, DS, MD, MDS, X10, X9, XS, A5, VA, VX, VC };

uint32_t encode(int p, int form, uint32_t xo) {
  const uint32_t base = uint32_t(p) << 26;
  switch (form) {
    case DS: return base | xo;
    case MD: case XS: return base | (xo << 2);
    case MDS: case X10: case X9: case A5: return base | (xo << 1);
    case VA: case VX: case VC: return base | xo;
    default: return base;
  }
}

// Checks that the encoding built from a table entry decodes back to that entry,
// including the OE (XO form) and record (VC form) variants.
void check_entry(ppu::Op op, const char* name, int p, int form, uint32_t xo, int flags) {
  char buf[160];
  const uint32_t raw = encode(p, form, xo);
  std::snprintf(buf, sizeof buf, "decode(%s) 0x%08X", name, raw);
  check(ppu::decode(raw).op == op, buf);
  if ((form == X9 && (flags & ppu::F_OE)) || form == VC) {
    std::snprintf(buf, sizeof buf, "decode(%s) variant 0x%08X", name, raw | (1u << 10));
    check(ppu::decode(raw | (1u << 10)).op == op, buf);
  }
}

void test_decoder() {
  using ppu::F_RC; using ppu::F_OE; using ppu::F_LK; using ppu::F_AA;
  std::printf("decoder table\n");
  int n = 0;
#define OP(e, m, f, fl, p, form, xo) check_entry(ppu::Op::e, m, p, form, xo, fl); ++n;
#include "ppu_ops.inc"
#undef OP
#define VOP(e, nm, k, form, xo) check_entry(ppu::Op::e, #nm, 4, form, xo, 0); ++n;
#include "ppu_vmx_ops.inc"
#undef VOP
  // a few known real encodings
  check(ppu::decode(0x7C0802A6).op == ppu::Op::Mfspr, "mflr r0");
  check(ppu::decode(0x4E800020).op == ppu::Op::Bclr, "blr");
  check(ppu::decode(0x786A1764).op == ppu::Op::Rldicr, "sldi r10,r3,2");
  check(ppu::disassemble(ppu::decode(0x786A1764), 0) == "rldicr r10,r3,2,61", "rldicr disasm");
  check(ppu::decode(0x7C6A07B4).op == ppu::Op::Extsw, "extsw r10,r3");
  check(ppu::decode(0x7C631E74).op == ppu::Op::Sradi, "sradi r3,r3,3");
  check(ppu::decode(0x60000000).op == ppu::Op::Ori, "nop");
  std::printf("  %d table entries checked\n", n);
}

// helpers to build vectors in guest element order
template <class T, class... A> ppu::v128 vec(A... a) {
  ppu::v128 v{};
  const T vals[] = {T(a)...};
  for (int i = 0; i < int(sizeof...(A)); ++i) ppu::vmx::set<T>(v, i, vals[i]);
  return v;
}
template <class T> T el(const ppu::v128& v, int i) { return ppu::vmx::get<T>(v, i); }

void test_vmx() {
  using namespace ppu::vmx;
  std::printf("vmx helpers\n");
  ppu::v128 d;

  vaddubs(d, vec<uint8_t>(250, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
          vec<uint8_t>(10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
  check(el<uint8_t>(d, 0) == 255 && el<uint8_t>(d, 1) == 2, "vaddubs saturates");

  vmrghw(d, vec<uint32_t>(1, 2, 3, 4), vec<uint32_t>(5, 6, 7, 8));
  check(el<uint32_t>(d, 0) == 1 && el<uint32_t>(d, 1) == 5 && el<uint32_t>(d, 2) == 2 && el<uint32_t>(d, 3) == 6, "vmrghw");
  vmrglw(d, vec<uint32_t>(1, 2, 3, 4), vec<uint32_t>(5, 6, 7, 8));
  check(el<uint32_t>(d, 0) == 3 && el<uint32_t>(d, 1) == 7 && el<uint32_t>(d, 2) == 4 && el<uint32_t>(d, 3) == 8, "vmrglw");

  vpkuwum(d, vec<uint32_t>(0x10001, 0x20002, 0x30003, 0x40004), vec<uint32_t>(5, 6, 7, 8));
  check(el<uint16_t>(d, 0) == 1 && el<uint16_t>(d, 3) == 4 && el<uint16_t>(d, 4) == 5 && el<uint16_t>(d, 7) == 8, "vpkuwum");
  vpkswss(d, vec<int32_t>(100000, -100000, 5, -5), vec<int32_t>(0, 0, 0, 0));
  check(el<int16_t>(d, 0) == 32767 && el<int16_t>(d, 1) == -32768 && el<int16_t>(d, 2) == 5, "vpkswss saturates");

  vupkhsb(d, vec<int8_t>(-1, 2, -3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
  check(el<int16_t>(d, 0) == -1 && el<int16_t>(d, 2) == -3 && el<int16_t>(d, 7) == 8, "vupkhsb");

  vsldoi(d, vec<uint8_t>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
         vec<uint8_t>(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31), 4);
  check(el<uint8_t>(d, 0) == 4 && el<uint8_t>(d, 11) == 15 && el<uint8_t>(d, 12) == 16 && el<uint8_t>(d, 15) == 19, "vsldoi");

  // vslo by 4 bytes == vsldoi with zeros
  vslo(d, vec<uint32_t>(1, 2, 3, 4), vec<uint8_t>(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 << 3));
  check(el<uint32_t>(d, 0) == 2 && el<uint32_t>(d, 2) == 4 && el<uint32_t>(d, 3) == 0, "vslo shifts left by octets");
  vsl(d, vec<uint32_t>(0x80000000u, 1, 0, 0), vec<uint8_t>(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
  check(el<uint32_t>(d, 0) == 0 && el<uint32_t>(d, 1) == 2, "vsl shifts across elements");

  vmsumshs(d, vec<int16_t>(1, 2, 3, 4, 32767, 32767, 0, 0), vec<int16_t>(5, 6, 7, 8, 32767, 32767, 0, 0),
           vec<int32_t>(100, 0x7FFFFFFF, 0, 0));
  check(el<int32_t>(d, 0) == 1 * 5 + 2 * 6 + 100 && el<int32_t>(d, 1) == 0x7FFFFFFF, "vmsumshs");

  vsum4ubs(d, vec<uint8_t>(1, 2, 3, 4, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0), vec<uint32_t>(10, 0xFFFFFFFFu, 0, 0));
  check(el<uint32_t>(d, 0) == 20 && el<uint32_t>(d, 1) == 0xFFFFFFFFu, "vsum4ubs");

  vctsxs(d, vec<float>(1.5f, -1e20f, 1e20f, std::bit_cast<float>(0x7FC00000u)), 0);
  check(el<int32_t>(d, 0) == 1 && el<int32_t>(d, 1) == INT32_MIN && el<int32_t>(d, 2) == INT32_MAX && el<int32_t>(d, 3) == 0,
        "vctsxs saturates, NaN -> 0");
  vcfsx(d, vec<int32_t>(8, -8, 0, 1), 2);
  check(el<float>(d, 0) == 2.0f && el<float>(d, 1) == -2.0f, "vcfsx scale");

  vmaddfp(d, vec<float>(2, 2, 2, 2), vec<float>(1, 1, 1, 1), vec<float>(3, 3, 3, 3));
  check(el<float>(d, 0) == 7.0f, "vmaddfp = a*c + b");
  vnmsubfp(d, vec<float>(2, 2, 2, 2), vec<float>(1, 1, 1, 1), vec<float>(3, 3, 3, 3));
  check(el<float>(d, 0) == -5.0f, "vnmsubfp = -(a*c - b)");

  vsel(d, vec<uint32_t>(0xFFFF0000u, 0, 0, 0), vec<uint32_t>(0x0000FFFFu, 0, 0, 0), vec<uint32_t>(0x00FF00FFu, 0, 0, 0));
  check(el<uint32_t>(d, 0) == 0xFF0000FFu, "vsel");

  vspltisw(d, -3);
  check(el<int32_t>(d, 0) == -3 && el<int32_t>(d, 3) == -3, "vspltisw");

  vcmpequw(d, vec<uint32_t>(1, 2, 3, 4), vec<uint32_t>(1, 0, 3, 0));
  check(el<uint32_t>(d, 0) == 0xFFFFFFFFu && el<uint32_t>(d, 1) == 0, "vcmpequw");
  const ppu::CRField cr = cr6_from(d);
  check(!cr.lt && !cr.eq, "cr6: mixed result");

  // lvlx | lvrx(ea+16) reconstructs an unaligned 16-byte load
  uint8_t mem[64];
  for (int i = 0; i < 64; ++i) mem[i] = uint8_t(i);
  ppu::v128 l, r;
  lvlx(l, mem, 5);
  lvrx(r, mem, 5 + 16);
  vor(d, l, r);
  bool ok = true;
  for (int i = 0; i < 16; ++i) ok &= el<uint8_t>(d, i) == 5 + i;
  check(ok, "lvlx | lvrx unaligned load");

  // stvlx + stvrx store an unaligned vector
  uint8_t out[64] = {};
  ppu::v128 s = vec<uint8_t>(100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115);
  stvlx(s, out, 5);
  stvrx(s, out, 5 + 16);
  ok = true;
  for (int i = 0; i < 16; ++i) ok &= out[5 + i] == 100 + i;
  check(ok && out[4] == 0 && out[21] == 0, "stvlx + stvrx unaligned store");
}

void test_scalar() {
  std::printf("scalar helpers\n");
  ppu::Context c;
  check(ppu::add3(c, ~0ull, 0, 1) == 0 && c.xer_ca == 1, "add3 carry from carry-in");
  check(ppu::add3(c, 5, 6, 0) == 11 && c.xer_ca == 0, "add3 no carry");
  check(ppu::srad(c, uint64_t(-1), 70) == ~0ull && c.xer_ca == 1, "srad >= 64 sign fill");
  check(ppu::mulh64(-1, 1) == ~0ull, "mulhd -1*1");
  check(ppu::umulh64(~0ull, 2) == 1, "mulhdu");
  check(ppu::divw(INT32_MIN, -1) == 0, "divw overflow defined as 0");
  check(ppu::fp_bits_lo(ppu::fctiw(-2.5, false)) == uint32_t(-2), "fctiw round-to-nearest-even");
  check(ppu::fp_bits_lo(ppu::fctiw(1e10, true)) == 0x7FFFFFFFu, "fctiwz saturates");
  check(ppu::fcfid(std::bit_cast<double>(uint64_t(-7))) == -7.0, "fcfid");
  ppu::set_cr(c, 0x12345678u, 0xFF);
  check(ppu::get_cr(c) == 0x12345678u, "mtcrf/mfcr round trip");
  ppu::set_xer(c, 0xE0000005ull);
  check(ppu::get_xer(c) == 0xE0000005ull, "mtxer/mfxer round trip");
  check(ppu::mask64(32 + 0, 32 + 29) == 0xFFFFFFFCull, "mask64 word mask");
  check(ppu::mask64(60, 3) == 0xF00000000000000Full, "mask64 wraps");
}

void test_rsx_fp() {
  std::printf("rsx fragment condition & alpha test\n");
  // Verify NV40 FP decoding matches Skate 3's CC write (SLT R63) and CC test (NE.yyyy).
  auto push_w = [](std::vector<uint8_t>& buf, uint32_t w) {
    buf.push_back(uint8_t(w >> 8));
    buf.push_back(uint8_t(w));
    buf.push_back(uint8_t(w >> 24));
    buf.push_back(uint8_t(w >> 16));
  };
  std::vector<uint8_t> raw;
  push_w(raw, 0x4A40037Eu); push_w(raw, 0x1C9DC918u); push_w(raw, 0x00000002u); push_w(raw, 0x0001C800u);
  push_w(raw, 0x00000000u); push_w(raw, 0x00000000u); push_w(raw, 0x00000000u); push_w(raw, 0x00000000u);  // dummy const
  push_w(raw, 0x04001803u); push_w(raw, 0x0AB48008u); push_w(raw, 0x00008002u); push_w(raw, 0x0002AA02u);
  push_w(raw, 0x00000000u); push_w(raw, 0x00000000u); push_w(raw, 0x00000000u); push_w(raw, 0x00000000u);
  const auto dec = rsx::decode_fp(raw.data());
  check(dec.size() == 2 && dec[0].op == rsx::FP_SLT && dec[0].set_cond && dec[0].no_dest && dec[0].mask == 1,
        "decode_fp SLT CC.x");
  check(dec[1].cond == 5 && dec[1].cond_swz[0] == 1 && dec[1].cond_swz[1] == 1 && dec[1].cond_swz[2] == 1 &&
            dec[1].cond_swz[3] == 1 && dec[1].end,
        "decode_fp NE.yyyy + END");

  auto csrc = [](float x, float y, float z, float w, rsx::FpInsn& i) {
    i.src[0].type = rsx::FP_CONST;
    i.has_constant = true;
    i.constant = {x, y, z, w};
  };
  std::vector<rsx::FpInsn> prog(6);
  // 0: R0 = {1, 2, 3, 4}
  prog[0].op = rsx::FP_MOV; prog[0].dst = 0; csrc(1, 2, 3, 4, prog[0]);
  // 1: SLT R63.x(set_cond, no_dest), IN0, 0
  prog[1].op = rsx::FP_SLT; prog[1].dst = 63; prog[1].mask = 1; prog[1].set_cond = true; prog[1].no_dest = true;
  prog[1].src[0].type = rsx::FP_INPUT; prog[1].input = rsx::FPI_COL0;
  prog[1].src[1].type = rsx::FP_CONST; prog[1].has_constant = true; prog[1].constant = {0, 0, 0, 0};
  // 2: SLT R63.y(set_cond, no_dest), 0.5, IN0 (leaves CC.x intact!)
  prog[2].op = rsx::FP_SLT; prog[2].dst = 63; prog[2].mask = 2; prog[2].set_cond = true; prog[2].no_dest = true;
  prog[2].src[0].type = rsx::FP_CONST; prog[2].has_constant = true; prog[2].constant = {0.5f, 0.5f, 0.5f, 0.5f};
  prog[2].src[1].type = rsx::FP_INPUT; prog[2].input = rsx::FPI_COL0;
  // 3: MOV R0.zw(NE.xxxx), {9, 9, 9, 9}
  prog[3].op = rsx::FP_MOV; prog[3].dst = 0; prog[3].mask = 0xC; prog[3].cond = 5;
  for (int k = 0; k < 4; ++k) prog[3].cond_swz[k] = 0;
  csrc(9, 9, 9, 9, prog[3]);
  // 4: MOV R0.xy(NE.yyyy), {7, 8, 7, 8}
  prog[4].op = rsx::FP_MOV; prog[4].dst = 0; prog[4].mask = 0x3; prog[4].cond = 5;
  for (int k = 0; k < 4; ++k) prog[4].cond_swz[k] = 1;
  csrc(7, 8, 7, 8, prog[4]);
  // 5: MOV R63.z(set_cond, no_dest), IN0; KIL(NE.zzzz); MOV R0.w, {5,5,5,5}
  std::vector<rsx::FpInsn> prog_k = prog;
  prog_k[5].op = rsx::FP_MOV; prog_k[5].dst = 63; prog_k[5].mask = 4; prog_k[5].set_cond = true;
  prog_k[5].no_dest = true; prog_k[5].end = false; prog_k[5].src[0].type = rsx::FP_INPUT; prog_k[5].input = rsx::FPI_COL0;
  rsx::FpInsn kil; kil.op = rsx::FP_KIL; kil.cond = 5;
  for (int k = 0; k < 4; ++k) kil.cond_swz[k] = 2;
  prog_k.push_back(kil);
  rsx::FpInsn tail; tail.op = rsx::FP_MOV; tail.dst = 0; tail.mask = 8; tail.end = true; csrc(5, 5, 5, 5, tail);
  prog_k.push_back(tail);

  rsx::vec4 in_neg[rsx::FPI_COUNT]{}, in_pos[rsx::FPI_COUNT]{}, in_kil[rsx::FPI_COUNT]{};
  in_neg[rsx::FPI_COL0] = {-1, -1, 0, 0};  // CC.x = 1 (NE), CC.y = 0 (EQ), CC.z = 0
  in_pos[rsx::FPI_COL0] = {1, 1, 0, 0};    // CC.x = 0 (EQ), CC.y = 1 (NE), CC.z = 0
  in_kil[rsx::FPI_COL0] = {1, 1, 2, 0};    // CC.z = 2 (NE -> KIL)
  rsx::FpResult r_neg, r_pos, r_kil;
  std::string err;
  check(rsx::run_fp(prog_k, in_neg, true, nullptr, nullptr, r_neg, &err) && !r_neg.discard &&
            r_neg.color[0] == 1.0f && r_neg.color[1] == 2.0f && r_neg.color[2] == 9.0f && r_neg.color[3] == 5.0f,
        "run_fp NE.xxxx writes R0.zw while CC.y=0 & continues past non-taken KIL");
  check(rsx::run_fp(prog_k, in_pos, true, nullptr, nullptr, r_pos, &err) && !r_pos.discard &&
            r_pos.color[0] == 7.0f && r_pos.color[1] == 8.0f && r_pos.color[2] == 3.0f && r_pos.color[3] == 5.0f,
        "run_fp NE.yyyy writes R0.xy while CC.x stays 0 & runs post-KIL MOV");
  check(rsx::run_fp(prog_k, in_kil, true, nullptr, nullptr, r_kil, &err) && r_kil.discard,
        "run_fp conditional KIL triggers when CC.z != 0");

  std::vector<uint32_t> spirv;
  check(rsx::spirv_translate_fp(prog_k, true, spirv, &err, true, 0x206, 0x80) && spirv.size() > 5 &&
            spirv[0] == 0x07230203u,
        "spirv_translate_fp conditional CC + conditional KIL + alpha test");

  // Verify OPDEST bit 21 (exp_tex / _bx2) expands texture fetches [0,1] -> [-1,1].
  std::vector<uint8_t> raw_bx2;
  // Real Skate 3 normal-map TEX instruction: 172A8E81 1C9DC801 0001C800 3FE1C800 (END set)
  push_w(raw_bx2, 0x172A8E81u); push_w(raw_bx2, 0x1C9DC801u); push_w(raw_bx2, 0x0001C800u); push_w(raw_bx2, 0x3FE1C800u);
  const auto dec_bx2 = rsx::decode_fp(raw_bx2.data());
  check(dec_bx2.size() == 1 && dec_bx2[0].op == rsx::FP_TEX && dec_bx2[0].exp_tex && dec_bx2[0].tex == 5,
        "decode_fp OPDEST bit 21 exp_tex (_bx2)");
  auto mid_tex = [](void*, uint32_t, const rsx::vec4&) -> rsx::vec4 { return {0.5f, 0.25f, 1.0f, 0.0f}; };
  rsx::FpResult r_bx2;
  rsx::vec4 in_bx2[rsx::FPI_COUNT]{};
  check(rsx::run_fp(dec_bx2, in_bx2, false, mid_tex, nullptr, r_bx2, &err) &&
            std::fabs(r_bx2.color[0] - 0.0f) < 1e-5f && std::fabs(r_bx2.color[1] - (-0.5f)) < 1e-5f &&
            std::fabs(r_bx2.color[2] - 1.0f) < 1e-5f,
        "run_fp exp_tex expands [0,1] -> [-1,1]");
  std::vector<uint32_t> spv_bx2;
  check(rsx::spirv_translate_fp(dec_bx2, false, spv_bx2, &err) && spv_bx2.size() > 5 && spv_bx2[0] == 0x07230203u,
        "spirv_translate_fp exp_tex");

  // Verify spirv_translate_vp produces a valid SPIR-V vertex shader with ARL/index_const + VLIW ops.
  rsx::VpState vp;
  vp.start = 0;
  vp.program[0].vec_op = rsx::VV_ARL;
  vp.program[0].src[0].type = rsx::VP_INPUT; vp.program[0].input = 3;
  vp.program[0].vec_mask = 1; vp.program[0].addr_reg = 0;
  vp.program[1].vec_op = rsx::VV_DP4; vp.program[1].sca_op = rsx::VS_RSQ;
  vp.program[1].src[0].type = rsx::VP_INPUT; vp.program[1].input = 0;
  vp.program[1].src[1].type = rsx::VP_CONST; vp.program[1].constant = 16; vp.program[1].index_const = true;
  vp.program[1].src[2].type = rsx::VP_INPUT; vp.program[1].input = 0;
  vp.program[1].dst_out = rsx::VPO_HPOS; vp.program[1].vec_result = true; vp.program[1].vec_mask = 0xF;
  vp.program[1].sca_dst_tmp = 1; vp.program[1].sca_mask = 1;
  vp.program[1].end = true;
  std::vector<uint32_t> vp_spv;
  rsx::SpirvVpMeta vp_meta;
  check(rsx::spirv_translate_vp(vp, vp_spv, vp_meta, &err) && vp_spv.size() > 5 && vp_spv[0] == 0x07230203u &&
            vp_meta.in_mask == ((1u << 0) | (1u << 3)) && vp_meta.in_count == 2 && vp_meta.uses_rel_const,
        "spirv_translate_vp ARL + index_const + VLIW");

  // End-to-end GPU vertex shader (draw_raw) vs CPU (prepare_draw + raster_cpu) pixel & culling parity test.
  if (auto gpu = rsx::GpuRenderer::create(nullptr)) {
    std::vector<uint8_t> mem(256 * 1024, 0);
    std::vector<uint32_t> regs(4096, 0);
    rsx::reset_defaults(regs.data());
    constexpr uint32_t kColorOff = 0x10000, kFpOff = 0x20000, kVtxOff = 0x30000;
    regs[rsx::m::SURFACE_FORMAT] = 8u | (2u << 5) | (1u << 8);  // A8R8G8B8 linear
    regs[rsx::m::SURFACE_PITCH_A] = 64 * 4;
    regs[rsx::m::SURFACE_COLOR_AOFFSET] = kColorOff;
    regs[rsx::m::SURFACE_CLIP_H] = (64u << 16);
    regs[rsx::m::SURFACE_CLIP_V] = (64u << 16);
    regs[rsx::m::VIEWPORT_SCALE + 0] = std::bit_cast<uint32_t>(32.0f);
    regs[rsx::m::VIEWPORT_SCALE + 1] = std::bit_cast<uint32_t>(-32.0f);
    regs[rsx::m::VIEWPORT_SCALE + 2] = std::bit_cast<uint32_t>(0.5f);
    regs[rsx::m::VIEWPORT_OFFSET + 0] = std::bit_cast<uint32_t>(32.0f);
    regs[rsx::m::VIEWPORT_OFFSET + 1] = std::bit_cast<uint32_t>(32.0f);
    regs[rsx::m::VIEWPORT_OFFSET + 2] = std::bit_cast<uint32_t>(0.5f);
    regs[rsx::m::SHADER_PROGRAM] = kFpOff | 1u;
    regs[rsx::m::SHADER_CONTROL] = 0x40;  // fp32 R0 output
    // Minimal FP at kFpOff: MOV R0, f[COL0] (END): op=1<<24, input=1<<13, mask=0xF<<9, end=1
    std::vector<uint8_t> fp_bytes;
    push_w(fp_bytes, 0x01003E01u);  // MOV R0.xyzw, input=COL0(1), END
    push_w(fp_bytes, 0x1C9DC801u);  // SRC0 = INPUT
    push_w(fp_bytes, 0x0001C800u);
    push_w(fp_bytes, 0x0001C800u);
    std::memcpy(mem.data() + kFpOff, fp_bytes.data(), fp_bytes.size());

    // 3 vertices (CCW on screen because vp_scale.y = -32 flips Y): (-1,-1,0,1), (1,-1,0,1), (-1,1,0,1)
    const float tri_pos[3][4] = {{-1.0f, -1.0f, 0.0f, 1.0f}, {1.0f, -1.0f, 0.0f, 1.0f}, {-1.0f, 1.0f, 0.0f, 1.0f}};
    uint8_t* vp_mem = mem.data() + kVtxOff;
    for (int v = 0; v < 3; ++v)
      for (int c = 0; c < 4; ++c) {
        const uint32_t u = std::bit_cast<uint32_t>(tri_pos[v][c]);
        vp_mem[(v * 4 + c) * 4 + 0] = uint8_t(u >> 24);
        vp_mem[(v * 4 + c) * 4 + 1] = uint8_t(u >> 16);
        vp_mem[(v * 4 + c) * 4 + 2] = uint8_t(u >> 8);
        vp_mem[(v * 4 + c) * 4 + 3] = uint8_t(u);
      }
    regs[rsx::m::VERTEX_ARRAY_OFFSET + 0] = kVtxOff;
    regs[rsx::m::VERTEX_ARRAY_FORMAT + 0] = 2u | (4u << 4) | (16u << 8);  // float4, stride 16

    rsx::VpState vp_draw;
    vp_draw.constants[5] = {64.0f / 255.0f, 128.0f / 255.0f, 192.0f / 255.0f, 1.0f};
    // insn 0: o[HPOS] = v[0]
    vp_draw.program[0].vec_op = rsx::VV_MOV;
    vp_draw.program[0].src[0].type = rsx::VP_INPUT; vp_draw.program[0].input = 0;
    vp_draw.program[0].dst_out = rsx::VPO_HPOS; vp_draw.program[0].vec_result = true; vp_draw.program[0].vec_mask = 0xF;
    // insn 1: o[COL0] = c[5] (END)
    vp_draw.program[1].vec_op = rsx::VV_MOV;
    vp_draw.program[1].src[0].type = rsx::VP_CONST; vp_draw.program[1].constant = 5;
    vp_draw.program[1].dst_out = rsx::VPO_COL0; vp_draw.program[1].vec_result = true; vp_draw.program[1].vec_mask = 0xF;
    vp_draw.program[1].end = true;

    rsx::vec4 const_att[16]{};
    for (auto& a : const_att) a = {0, 0, 0, 1};
    rsx::DrawEnv env{mem.data(), regs.data(), &vp_draw, const_att, [](uint32_t off, uint32_t) { return off; }};
    rsx::DrawCall call{5 /*TRIANGLES*/, false, {{0, 3}}};

    // 1. CPU reference render with back-face culling (front=CCW, tri is CCW -> kept).
    regs[rsx::m::CULL_FACE_ENABLE] = 1;
    regs[rsx::m::CULL_FACE] = 0x405;   // BACK
    regs[rsx::m::FRONT_FACE] = 0x901;  // CCW
    rsx::PreparedDraw pd_cpu;
    rsx::DrawStats ds;
    check(rsx::prepare_draw(env, call, pd_cpu, &err) && pd_cpu.tris.size() == 3 &&
              rsx::raster_cpu(env, pd_cpu, ds, &err),
          "CPU prepare_draw + raster_cpu reference");
    const uint32_t cpu_px = *reinterpret_cast<const uint32_t*>(mem.data() + kColorOff + (40 * 64 + 16) * 4);
    check(cpu_px != 0, "CPU rendered non-zero pixel inside CCW triangle");

    // 2. GPU draw_raw render of the exact same state -> must match CPU pixel!
    std::memset(mem.data() + kColorOff, 0, 64 * 64 * 4);
    gpu->invalidate(kColorOff, 64 * 64 * 4);
    rsx::PreparedDraw pd_gpu;
    uint32_t gpu_tris = 0;
    check(rsx::prepare_draw_state(env, pd_gpu, &err) &&
              gpu->draw_raw(env, call, pd_gpu, gpu_tris, &err) && gpu_tris == 1,
          "GPU draw_raw executes translated VP on Vulkan");
    gpu->flush(mem.data(), false);
    const uint32_t gpu_px = *reinterpret_cast<const uint32_t*>(mem.data() + kColorOff + (40 * 64 + 16) * 4);
    check(gpu_px == cpu_px, "GPU draw_raw pixel matches CPU raster_cpu pixel");

    // 3. Flip FRONT_FACE to CW (0x900) so the CCW triangle is back-facing and culled on both CPU and GPU.
    regs[rsx::m::FRONT_FACE] = 0x900;  // CW
    std::memset(mem.data() + kColorOff, 0, 64 * 64 * 4);
    gpu->invalidate(kColorOff, 64 * 64 * 4);
    check(rsx::prepare_draw_state(env, pd_gpu, &err) &&
              gpu->draw_raw(env, call, pd_gpu, gpu_tris, &err),
          "GPU draw_raw with FRONT_FACE=CW back-face cull");
    gpu->flush(mem.data(), false);
    const uint32_t culled_px = *reinterpret_cast<const uint32_t*>(mem.data() + kColorOff + (40 * 64 + 16) * 4);
    check(culled_px == 0, "GPU draw_raw culls back-facing triangle matching CPU winding");

    // 4. With FRONT_FACE=CW (0x900) and CULL_FACE=FRONT (0x404), the back-facing triangle is KEPT.
    regs[rsx::m::CULL_FACE] = 0x404;  // FRONT
    std::memset(mem.data() + kColorOff, 0, 64 * 64 * 4);
    gpu->invalidate(kColorOff, 64 * 64 * 4);
    check(rsx::prepare_draw_state(env, pd_gpu, &err) &&
              gpu->draw_raw(env, call, pd_gpu, gpu_tris, &err) && gpu_tris == 1,
          "GPU draw_raw with FRONT_FACE=CW + CULL_FACE=FRONT");
    gpu->flush(mem.data(), false);
    const uint32_t kept_px = *reinterpret_cast<const uint32_t*>(mem.data() + kColorOff + (40 * 64 + 16) * 4);
    check(kept_px == cpu_px, "GPU draw_raw keeps back-facing triangle when CULL_FACE=FRONT");
  }
}

}  // namespace

int main() {
  test_decoder();
  test_vmx();
  test_scalar();
  test_rsx_fp();
  if (g_fail) std::printf("%d of %d checks FAILED\n", g_fail, g_checks);
  else std::printf("all %d checks passed\n", g_checks);
  return g_fail ? 1 : 0;
}
