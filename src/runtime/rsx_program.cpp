#include "rsx_program.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rsx {
namespace {

// ---- decoding helpers ----------------------------------------------------------
VpSrc vp_src(uint32_t v, bool abs) {
  VpSrc s;
  s.type = uint8_t(v & 3);
  s.index = uint8_t((v >> 2) & 0x3F);
  s.swz[3] = uint8_t((v >> 8) & 3);
  s.swz[2] = uint8_t((v >> 10) & 3);
  s.swz[1] = uint8_t((v >> 12) & 3);
  s.swz[0] = uint8_t((v >> 14) & 3);
  s.neg = (v >> 16) & 1;
  s.abs = abs;
  return s;
}

FpSrc fp_src(uint32_t v, bool abs) {
  FpSrc s;
  s.type = uint8_t(v & 3);
  s.index = uint8_t((v >> 2) & 0x3F);
  s.fp16 = (v >> 8) & 1;
  s.swz[0] = uint8_t((v >> 9) & 3);
  s.swz[1] = uint8_t((v >> 11) & 3);
  s.swz[2] = uint8_t((v >> 13) & 3);
  s.swz[3] = uint8_t((v >> 15) & 3);
  s.neg = (v >> 17) & 1;
  s.abs = abs;
  return s;
}

// FP words are stored big-endian with their two 16-bit halves swapped. (Verified
// against shaders from real compiled PS3 programs: e.g. "MOV R0, f[COL0]" is stored
// as bytes 3E 01 01 00 for the instruction word 0x01003E01.)
uint32_t fp_word(const uint8_t* p) {
  const uint32_t be = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
  return (be << 16) | (be >> 16);
}

// ---- math helpers --------------------------------------------------------------------
vec4 splat(float f) { return {f, f, f, f}; }
float dot3(const vec4& a, const vec4& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
float dot4(const vec4& a, const vec4& b) { return dot3(a, b) + a[3] * b[3]; }
template <class F> vec4 map2(const vec4& a, const vec4& b, F f) {
  return {f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])};
}
template <class F> vec4 map1(const vec4& a, F f) { return {f(a[0]), f(a[1]), f(a[2]), f(a[3])}; }
float sat(float f) { return std::clamp(f, 0.0f, 1.0f); }
float fract(float f) { return f - std::floor(f); }
float rsq(float f) { return 1.0f / std::sqrt(std::max(std::fabs(f), 1e-38f)); }

vec4 lit(const vec4& s) {
  // (1, max(x,0), x > 0 ? max(y,0)^clamp(w,-128,128) : 0, 1)
  const float x = std::max(s[0], 0.0f), y = std::max(s[1], 0.0f), w = std::clamp(s[3], -128.0f, 128.0f);
  return {1.0f, x, s[0] > 0.0f ? std::pow(y, w) : 0.0f, 1.0f};
}

template <class Src>
vec4 apply_modifiers(const vec4& v, const Src& s) {
  vec4 r{v[s.swz[0]], v[s.swz[1]], v[s.swz[2]], v[s.swz[3]]};
  if (s.abs) r = map1(r, [](float f) { return std::fabs(f); });
  if (s.neg) r = map1(r, [](float f) { return -f; });
  return r;
}

void write_masked(vec4& dst, const vec4& v, uint8_t mask) {
  for (int c = 0; c < 4; ++c)
    if (mask & (1 << c)) dst[c] = v[c];
}

const char* const kVpVecNames[] = {"NOP", "MOV", "MUL", "ADD", "MAD", "DP3", "DPH", "DP4", "DST", "MIN", "MAX",
                                   "SLT", "SGE", "ARL", "FRC", "FLR", "SEQ", "SFL", "SGT", "SLE", "SNE", "STR",
                                   "SSG", "?17", "?18", "TXL"};
const char* const kVpScaNames[] = {"NOP", "MOV", "RCP", "RCC", "RSQ", "EXP", "LOG", "LIT", "BRA", "BRI", "CAL",
                                   "CLI", "RET", "LG2", "EX2", "SIN", "COS", "BRB", "CLB", "PSH", "POP"};

}  // namespace

// ============================================================================
// Vertex programs
// ============================================================================
VpInsn decode_vp(const uint32_t w[4]) {
  const uint32_t d0 = w[0], d1 = w[1], d2 = w[2], d3 = w[3];
  VpInsn i;
  i.cond_test = (d0 >> 13) & 1;
  i.dst_tmp = uint8_t((d0 >> 15) & 0x3F);
  i.saturate = (d0 >> 26) & 1;
  i.vec_result = (d0 >> 30) & 1;
  i.input = uint8_t((d1 >> 8) & 0xF);
  i.constant = uint16_t((d1 >> 12) & 0x3FF);
  i.vec_op = uint8_t((d1 >> 22) & 0x1F);
  i.sca_op = uint8_t((d1 >> 27) & 0x1F);
  const uint32_t s0 = ((d1 & 0xFF) << 9) | (d2 >> 23);
  const uint32_t s1 = (d2 >> 6) & 0x1FFFF;
  const uint32_t s2 = ((d2 & 0x3F) << 11) | (d3 >> 21);
  i.src[0] = vp_src(s0, (d0 >> 21) & 1);
  i.src[1] = vp_src(s1, (d0 >> 22) & 1);
  i.src[2] = vp_src(s2, (d0 >> 23) & 1);
  i.end = d3 & 1;
  i.dst_out = uint8_t((d3 >> 2) & 0x1F);
  i.sca_dst_tmp = uint8_t((d3 >> 7) & 0x3F);
  i.vec_mask = uint8_t(((d3 >> 16) & 1) | (((d3 >> 15) & 1) << 1) | (((d3 >> 14) & 1) << 2) | (((d3 >> 13) & 1) << 3));
  i.sca_mask = uint8_t(((d3 >> 20) & 1) | (((d3 >> 19) & 1) << 1) | (((d3 >> 18) & 1) << 2) | (((d3 >> 17) & 1) << 3));
  // D0: addr_swz 0-1, cond swizzle 2-9 (w,z,y,x), cond 10-12, cond_test 13,
  // cond_update 14 (+29), addr_reg_sel 24, cond_reg_sel 25. D3: index_const 1.
  i.addr_swz = uint8_t(d0 & 3);
  i.cond_swz[3] = uint8_t((d0 >> 2) & 3);
  i.cond_swz[2] = uint8_t((d0 >> 4) & 3);
  i.cond_swz[1] = uint8_t((d0 >> 6) & 3);
  i.cond_swz[0] = uint8_t((d0 >> 8) & 3);
  i.cond = uint8_t((d0 >> 10) & 7);
  i.cond_update = ((d0 >> 14) & 1) && ((d0 >> 29) & 1);
  i.addr_reg = uint8_t((d0 >> 24) & 1);
  i.cond_reg = uint8_t((d0 >> 25) & 1);
  i.index_const = (d3 >> 1) & 1;
  // Flow control target: D2 bits 0-5 (high) and D3 bits 29-31 (low).
  i.branch_target = uint16_t(((d2 & 0x3F) << 3) | ((d3 >> 29) & 7));
  return i;
}

bool run_vp(const VpState& vp, const vec4 inputs[16], vec4 out[VPO_COUNT], std::string* error) {
  vec4 tmp[64]{};
  for (int i = 0; i < VPO_COUNT; ++i) out[i] = vec4{0, 0, 0, 0};
  out[VPO_HPOS] = {0, 0, 0, 1};
  int areg[2][4] = {};              // address registers A0, A1
  vec4 cc[2] = {};                  // condition registers CC0, CC1
  uint32_t call_stack[8];
  int sp = 0;

  // Instruction budget guards against runaway loops in malformed programs.
  for (uint32_t pc = vp.start, n = 0; pc < vp.program.size() && n < 4096; ++pc, ++n) {
    const VpInsn& i = vp.program[pc];
    auto src = [&](int k) {
      const VpSrc& s = i.src[k];
      vec4 v{0, 0, 0, 0};
      if (s.type == VP_TEMP) v = tmp[s.index];
      else if (s.type == VP_INPUT) v = inputs[i.input];
      else if (s.type == VP_CONST) {
        int idx = i.constant;
        if (i.index_const) idx += areg[i.addr_reg][i.addr_swz];
        const int n_const = int(vp.constants.size());
        v = vp.constants[size_t(((idx % n_const) + n_const) % n_const)];
      }
      return apply_modifiers(v, s);
    };
    // Condition test for component c (CC compared against zero).
    auto cond_ok = [&](int c) {
      if (i.cond == 7) return true;
      if (i.cond == 0) return false;
      const float v = cc[i.cond_reg][i.cond_swz[c]];
      return ((i.cond & 1) && v < 0.0f) || ((i.cond & 2) && v == 0.0f) || ((i.cond & 4) && v > 0.0f);
    };
    // Write mask restricted by the condition (when conditional writes are enabled).
    auto effective_mask = [&](uint8_t mask) {
      if (!i.cond_test) return mask;
      uint8_t m = 0;
      for (int c = 0; c < 4; ++c)
        if ((mask & (1 << c)) && cond_ok(c)) m |= uint8_t(1 << c);
      return m;
    };

    // Fetch all sources and condition masks before either VLIW slot writes back.
    const vec4 a = src(0), b = src(1), c = src(2);
    const uint8_t vm = effective_mask(i.vec_mask);
    const uint8_t sm = effective_mask(i.sca_mask);
    const bool br_cond = cond_ok(0);

    if (i.vec_op == VV_ARL) {
      // Address register load: A = floor(src0) for the written components.
      for (int k = 0; k < 4; ++k)
        if (vm & (1 << k)) areg[i.addr_reg][k] = int(std::floor(a[k]));
    } else if (i.vec_op != VV_NOP) {
      vec4 r{};
      switch (i.vec_op) {
        case VV_MOV: r = a; break;
        case VV_MUL: r = map2(a, b, [](float x, float y) { return x * y; }); break;
        case VV_ADD: r = map2(a, c, [](float x, float y) { return x + y; }); break;  // ADD reads src0 + src2
        case VV_MAD: r = {a[0] * b[0] + c[0], a[1] * b[1] + c[1], a[2] * b[2] + c[2], a[3] * b[3] + c[3]}; break;
        case VV_DP3: r = splat(dot3(a, b)); break;
        case VV_DPH: r = splat(dot3(a, b) + b[3]); break;
        case VV_DP4: r = splat(dot4(a, b)); break;
        case VV_DST: r = {1.0f, a[1] * b[1], a[2], b[3]}; break;
        case VV_MIN: r = map2(a, b, [](float x, float y) { return std::min(x, y); }); break;
        case VV_MAX: r = map2(a, b, [](float x, float y) { return std::max(x, y); }); break;
        case VV_SLT: r = map2(a, b, [](float x, float y) { return x < y ? 1.0f : 0.0f; }); break;
        case VV_SGE: r = map2(a, b, [](float x, float y) { return x >= y ? 1.0f : 0.0f; }); break;
        case VV_SEQ: r = map2(a, b, [](float x, float y) { return x == y ? 1.0f : 0.0f; }); break;
        case VV_SGT: r = map2(a, b, [](float x, float y) { return x > y ? 1.0f : 0.0f; }); break;
        case VV_SLE: r = map2(a, b, [](float x, float y) { return x <= y ? 1.0f : 0.0f; }); break;
        case VV_SNE: r = map2(a, b, [](float x, float y) { return x != y ? 1.0f : 0.0f; }); break;
        case VV_SFL: r = splat(0.0f); break;
        case VV_STR: r = splat(1.0f); break;
        case VV_FRC: r = map1(a, fract); break;
        case VV_FLR: r = map1(a, [](float x) { return std::floor(x); }); break;
        case VV_SSG: r = map1(a, [](float x) { return float((x > 0) - (x < 0)); }); break;
        default:
          if (error) *error = std::string("unsupported vertex vector op ") + kVpVecNames[std::min<int>(i.vec_op, 25)];
          return false;
      }
      r = map1(r, [](float x) { return std::isfinite(x) ? x : 0.0f; });
      if (i.saturate) r = map1(r, sat);
      if (i.dst_tmp != 0x3F) write_masked(tmp[i.dst_tmp], r, vm);
      if (i.vec_result && i.dst_out < VPO_COUNT) write_masked(out[i.dst_out], r, vm);
      if (i.cond_update) write_masked(cc[i.cond_reg], r, vm);
    }

    // Flow control lives in the scalar slot.
    if (i.sca_op == VS_BRA || i.sca_op == VS_BRI || i.sca_op == VS_CAL || i.sca_op == VS_CLI ||
        i.sca_op == VS_RET || i.sca_op == VS_BRB || i.sca_op == VS_CLB) {
      const bool taken = (i.sca_op == VS_BRB || i.sca_op == VS_CLB) ? false  // boolean-constant branches: not modelled
                         : (i.sca_op == VS_CAL) ? true
                         : br_cond;
      if (i.sca_op == VS_RET) {
        if (taken) {
          if (sp == 0) return true;  // return from the main program ends it
          pc = call_stack[--sp] - 1;
        }
      } else if (taken) {
        if (i.sca_op == VS_CAL || i.sca_op == VS_CLI || i.sca_op == VS_CLB) {
          if (sp == 8) {
            if (error) *error = "vertex program call stack overflow";
            return false;
          }
          call_stack[sp++] = pc + 1;
        }
        pc = uint32_t(i.branch_target) - 1;  // the loop increments
        continue;
      }
      if (i.end) return true;
      continue;
    }

    if (i.sca_op != VS_NOP) {
      const vec4& s = c;
      vec4 r{};
      switch (i.sca_op) {
        case VS_MOV: r = s; break;
        case VS_RCP: r = splat(s[0] != 0.0f ? 1.0f / s[0] : 0.0f); break;
        case VS_RCC: r = splat(std::clamp(s[0] != 0.0f ? 1.0f / s[0] : 0.0f, 5.42101e-20f, 1.884467e19f)); break;
        case VS_RSQ: r = splat(rsq(s[0])); break;
        case VS_EXP: {
          const float x = std::clamp(s[0], -126.0f, 126.0f);
          r = {std::exp2(std::floor(x)), fract(x), std::exp2(x), 1.0f};
          break;
        }
        case VS_LOG: {
          const float x = std::max(std::fabs(s[0]), 1e-38f);
          const float e = std::floor(std::log2(x));
          r = {e, x / std::exp2(e), std::log2(x), 1.0f};
          break;
        }
        case VS_LIT: r = lit(s); break;
        case VS_LG2: r = splat(std::log2(std::max(std::fabs(s[0]), 1e-38f))); break;
        case VS_EX2: r = splat(std::exp2(std::clamp(s[0], -126.0f, 126.0f))); break;
        case VS_SIN: r = splat(std::sin(s[0])); break;
        case VS_COS: r = splat(std::cos(s[0])); break;
        default:
          if (error) *error = std::string("unsupported vertex scalar op ") + kVpScaNames[std::min<int>(i.sca_op, 20)];
          return false;
      }
      r = map1(r, [](float x) { return std::isfinite(x) ? x : 0.0f; });
      if (i.saturate) r = map1(r, sat);
      if (i.sca_dst_tmp != 0x3F) write_masked(tmp[i.sca_dst_tmp], r, sm);
      if (!i.vec_result && i.dst_out < VPO_COUNT) write_masked(out[i.dst_out], r, sm);
      if (i.cond_update && i.vec_op == VV_NOP) write_masked(cc[i.cond_reg], r, sm);
    }
    if (i.end) return true;
  }
  return true;
}

// ============================================================================
// Fragment programs
// ============================================================================
std::vector<FpInsn> decode_fp(const uint8_t* p, size_t max_insns) {
  std::vector<FpInsn> prog;
  for (size_t n = 0; n < max_insns; ++n) {
    const uint32_t dst = fp_word(p), s0 = fp_word(p + 4), s1 = fp_word(p + 8), s2 = fp_word(p + 12);
    FpInsn i;
    i.end = dst & 1;
    i.dst = uint8_t((dst >> 1) & 0x3F);
    i.dst_fp16 = (dst >> 7) & 1;
    i.set_cond = (dst >> 8) & 1;
    i.mask = uint8_t((dst >> 9) & 0xF);
    i.input = uint8_t((dst >> 13) & 0xF);
    i.tex = uint8_t((dst >> 17) & 0xF);
    i.exp_tex = (dst >> 21) & 1;
    i.op = uint8_t(((dst >> 24) & 0x3F) | (((s1 >> 31) & 1) << 6));
    i.no_dest = (dst >> 30) & 1;
    i.saturate = (dst >> 31) & 1;
    i.scale = uint8_t((s1 >> 28) & 7);
    i.cond = uint8_t((s0 >> 18) & 7);
    for (int c = 0; c < 4; ++c) i.cond_swz[c] = uint8_t((s0 >> (21 + 2 * c)) & 3);
    i.src[0] = fp_src(s0, (s0 >> 29) & 1);
    i.src[1] = fp_src(s1, (s1 >> 18) & 1);
    i.src[2] = fp_src(s2, (s2 >> 18) & 1);
    p += 16;
    for (const FpSrc& s : i.src) i.has_constant |= s.type == FP_CONST;
    if (i.has_constant) {
      for (int c = 0; c < 4; ++c) i.constant[c] = std::bit_cast<float>(fp_word(p + 4 * c));
      p += 16;
    }
    prog.push_back(i);
    if (i.end) break;
  }
  return prog;
}

bool fp_cond_pass(uint8_t cond, float v) {
  switch (cond) {
    case 0: return false;
    case 1: return v < 0.0f;
    case 2: return v == 0.0f;
    case 3: return v <= 0.0f;
    case 4: return v > 0.0f;
    case 5: return v != 0.0f;
    case 6: return v >= 0.0f;
    default: return true;
  }
}

bool run_fp(const std::vector<FpInsn>& prog, const vec4 inputs[FPI_COUNT], bool fp32_output, TexSampler tex,
            void* tex_user, FpResult& out, std::string* error) {
  vec4 R[64]{}, H[128]{}, cc{};
  bool h0_written = false;
  out.discard = false;

  for (const FpInsn& i : prog) {
    auto src = [&](int k) {
      const FpSrc& s = i.src[k];
      vec4 v{0, 0, 0, 0};
      if (s.type == FP_TEMP) v = s.fp16 ? H[s.index] : R[s.index];
      else if (s.type == FP_INPUT) v = inputs[i.input < FPI_COUNT ? i.input : 0];
      else if (s.type == FP_CONST) v = i.constant;
      return apply_modifiers(v, s);
    };
    if (i.op == FP_FENCT || i.op == FP_FENCB || i.op == FP_NOP) {  // scheduling fences / nop: no effect on results
      if (i.end) break;
      continue;
    }
    bool cpass[4] = {true, true, true, true};
    if (i.cond != 7) {
      for (int k = 0; k < 4; ++k) cpass[k] = fp_cond_pass(i.cond, cc[i.cond_swz[k]]);
    }
    if (i.op == FP_KIL) {
      if (cpass[0] || cpass[1] || cpass[2] || cpass[3]) {
        out.discard = true;
        return true;
      }
      if (i.end) break;
      continue;
    }
    const vec4 a = src(0), b = src(1), c = src(2);
    vec4 r{};
    switch (i.op) {
      case FP_MOV: r = a; break;
      case FP_MUL: r = map2(a, b, [](float x, float y) { return x * y; }); break;
      case FP_ADD: r = map2(a, b, [](float x, float y) { return x + y; }); break;
      case FP_MAD: r = {a[0] * b[0] + c[0], a[1] * b[1] + c[1], a[2] * b[2] + c[2], a[3] * b[3] + c[3]}; break;
      case FP_DP3: r = splat(dot3(a, b)); break;
      case FP_DP4: r = splat(dot4(a, b)); break;
      case FP_DP2: r = splat(a[0] * b[0] + a[1] * b[1]); break;
      case FP_DP2A: r = splat(a[0] * b[0] + a[1] * b[1] + c[0]); break;
      case FP_DST: r = {1.0f, a[1] * b[1], a[2], b[3]}; break;
      case FP_MIN: r = map2(a, b, [](float x, float y) { return std::min(x, y); }); break;
      case FP_MAX: r = map2(a, b, [](float x, float y) { return std::max(x, y); }); break;
      case FP_SLT: r = map2(a, b, [](float x, float y) { return x < y ? 1.0f : 0.0f; }); break;
      case FP_SGE: r = map2(a, b, [](float x, float y) { return x >= y ? 1.0f : 0.0f; }); break;
      case FP_SLE: r = map2(a, b, [](float x, float y) { return x <= y ? 1.0f : 0.0f; }); break;
      case FP_SGT: r = map2(a, b, [](float x, float y) { return x > y ? 1.0f : 0.0f; }); break;
      case FP_SNE: r = map2(a, b, [](float x, float y) { return x != y ? 1.0f : 0.0f; }); break;
      case FP_SEQ: r = map2(a, b, [](float x, float y) { return x == y ? 1.0f : 0.0f; }); break;
      case FP_SFL: r = splat(0.0f); break;
      case FP_STR: r = splat(1.0f); break;
      case FP_FRC: r = map1(a, fract); break;
      case FP_FLR: r = map1(a, [](float x) { return std::floor(x); }); break;
      case FP_DDX: case FP_DDY: r = splat(0.0f); break;  // derivatives need pixel quads (GPU backend)
      case FP_TEX: case FP_TXL: case FP_TXB: case FP_TXD:
        r = tex ? tex(tex_user, i.tex, a) : vec4{0, 0, 0, 1};
        if (i.exp_tex) r = map1(r, [](float x) { return x * 2.0f - 1.0f; });
        break;
      case FP_TXP: {
        const float w = a[3] != 0.0f ? a[3] : 1.0f;
        r = tex ? tex(tex_user, i.tex, vec4{a[0] / w, a[1] / w, a[2] / w, 1.0f}) : vec4{0, 0, 0, 1};
        if (i.exp_tex) r = map1(r, [](float x) { return x * 2.0f - 1.0f; });
        break;
      }
      case FP_RCP: r = splat(a[0] != 0.0f ? 1.0f / a[0] : 0.0f); break;
      case FP_RSQ: r = splat(rsq(a[0])); break;
      case FP_EX2: r = splat(std::exp2(std::clamp(a[0], -126.0f, 126.0f))); break;
      case FP_LG2: r = splat(std::log2(std::max(std::fabs(a[0]), 1e-38f))); break;
      case FP_LIT: r = lit(a); break;
      case FP_LRP: r = {a[0] * b[0] + (1 - a[0]) * c[0], a[1] * b[1] + (1 - a[1]) * c[1],
                        a[2] * b[2] + (1 - a[2]) * c[2], a[3] * b[3] + (1 - a[3]) * c[3]}; break;
      case FP_COS: r = splat(std::cos(a[0])); break;
      case FP_SIN: r = splat(std::sin(a[0])); break;
      case FP_POW: r = splat(std::pow(std::max(std::fabs(a[0]), 1e-20f), std::clamp(b[0], -126.0f, 126.0f))); break;
      case FP_NRM: {
        const float len = std::sqrt(dot3(a, a));
        r = len > 0 ? vec4{a[0] / len, a[1] / len, a[2] / len, a[3]} : a;
        break;
      }
      case FP_DIV: r = map1(a, [&](float x) { return b[0] != 0.0f ? x / b[0] : 0.0f; }); break;
      case FP_DIVSQ: r = map1(a, [&](float x) { return x * rsq(b[0]); }); break;
      case FP_REFL: {
        const float d = 2.0f * dot3(a, b) / std::max(dot3(a, a), 1e-20f);
        r = {d * a[0] - b[0], d * a[1] - b[1], d * a[2] - b[2], b[3]};
        break;
      }
      default: {
        if (error) {
          char buf[64];
          std::snprintf(buf, sizeof buf, "unsupported fragment op %u", unsigned(i.op));
          *error = buf;
        }
        return false;
      }
    }

    if (!i.no_dest || i.set_cond) {
      static const float kScale[8] = {1, 2, 4, 8, 1, 0.5f, 0.25f, 0.125f};
      if (i.scale) r = map1(r, [&](float x) { return x * kScale[i.scale]; });
      if (i.saturate) r = map1(r, sat);
      uint8_t wmask = 0;
      for (int k = 0; k < 4; ++k)
        if ((i.mask & (1 << k)) && cpass[k]) wmask |= uint8_t(1 << k);
      if (wmask) {
        if (!i.no_dest) {
          if (i.dst_fp16) {
            write_masked(H[i.dst], r, wmask);
            if (i.dst == 0) h0_written = true;
          } else {
            write_masked(R[i.dst], r, wmask);
          }
        }
        if (i.set_cond) write_masked(cc, r, wmask);
      }
    }
    if (i.end) break;
  }
  out.color = (fp32_output || !h0_written) ? R[0] : H[0];
  return true;
}

std::string disasm_vp(const VpInsn& i) {
  char buf[128];
  std::snprintf(buf, sizeof buf, "%s/%s tmp=%d out=%d sca_tmp=%d in=%d c=%d mask=%X/%X%s", kVpVecNames[std::min<int>(i.vec_op, 25)],
                kVpScaNames[std::min<int>(i.sca_op, 20)], i.dst_tmp == 0x3F ? -1 : i.dst_tmp,
                i.dst_out == 0x1F ? -1 : i.dst_out, i.sca_dst_tmp == 0x3F ? -1 : i.sca_dst_tmp, i.input, i.constant,
                i.vec_mask, i.sca_mask, i.end ? " END" : "");
  return buf;
}

std::string disasm_fp(const FpInsn& i) {
  static constexpr const char* kCond[8] = {"FL", "LT", "EQ", "LE", "GT", "NE", "GE", "TR"};
  static constexpr char kSwz[4] = {'x', 'y', 'z', 'w'};
  char cbuf[16] = "";
  if (i.cond != 7 || i.cond_swz[0] != 0 || i.cond_swz[1] != 1 || i.cond_swz[2] != 2 || i.cond_swz[3] != 3) {
    std::snprintf(cbuf, sizeof cbuf, "(%s.%c%c%c%c)", kCond[i.cond & 7], kSwz[i.cond_swz[0]], kSwz[i.cond_swz[1]],
                  kSwz[i.cond_swz[2]], kSwz[i.cond_swz[3]]);
  }
  char buf[144];
  std::snprintf(buf, sizeof buf, "op%u%s%s%s %c%u mask=%X in=%u tex=%u%s%s%s", unsigned(i.op), i.exp_tex ? "_bx2" : "",
                i.set_cond ? "C" : "", cbuf, i.dst_fp16 ? 'H' : 'R', i.dst, i.mask, i.input, i.tex,
                i.no_dest ? " nodst" : "", i.has_constant ? " +const" : "", i.end ? " END" : "");
  return buf;
}

}  // namespace rsx
