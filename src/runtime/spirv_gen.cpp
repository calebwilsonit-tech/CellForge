#include "spirv_gen.h"

#include <bit>
#include <cstring>
#include <map>
#include <set>

namespace rsx {
namespace {

// ---- SPIR-V constants used here ----------------------------------------------------
enum : uint32_t {
  OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
  OpCapability = 17, OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
  OpTypeImage = 25, OpTypeSampledImage = 27, OpTypeArray = 28, OpTypeStruct = 30, OpTypePointer = 32,
  OpTypeFunction = 33, OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44,
  OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
  OpDecorate = 71, OpMemberDecorate = 72, OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81,
  OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88, OpImageSampleDrefImplicitLod = 89,
  OpImageSampleDrefExplicitLod = 90, OpConvertFToS = 110, OpFNegate = 127,
  OpIAdd = 128, OpFAdd = 129, OpFSub = 131, OpFMul = 133, OpFDiv = 136, OpVectorTimesScalar = 142,
  OpDot = 148, OpAny = 154, OpLogicalAnd = 167, OpSelect = 169, OpFOrdEqual = 180, OpFOrdNotEqual = 182,
  OpFUnordNotEqual = 183, OpFOrdLessThan = 184, OpFOrdGreaterThan = 186, OpFOrdLessThanEqual = 188,
  OpFOrdGreaterThanEqual = 190, OpBitwiseAnd = 199, OpSelectionMerge = 247,
  OpLabel = 248, OpBranch = 249, OpBranchConditional = 250, OpKill = 252, OpReturn = 253,
};
enum : uint32_t { SC_UniformConstant = 0, SC_Input = 1, SC_Uniform = 2, SC_Output = 3, SC_Function = 7 };
enum : uint32_t {
  Dec_Block = 2, Dec_ArrayStride = 6, Dec_BuiltIn = 11, Dec_Location = 30, Dec_Binding = 33, Dec_DescriptorSet = 34,
  Dec_Offset = 35
};
enum : uint32_t { BI_Position = 0, BI_FragCoord = 15 };
// GLSL.std.450
enum : uint32_t {
  G_FAbs = 4, G_FSign = 6, G_Floor = 8, G_Fract = 10, G_Sin = 13, G_Cos = 14, G_Pow = 26, G_Exp2 = 29, G_Log2 = 30,
  G_Sqrt = 31, G_InverseSqrt = 32, G_FMin = 37, G_FMax = 40, G_FClamp = 43, G_FMix = 46, G_Normalize = 69,
};

// ---- a minimal module builder ----------------------------------------------------------
class Module {
 public:
  uint32_t id() { return next_++; }

  static void emit(std::vector<uint32_t>& s, uint32_t op, const std::vector<uint32_t>& args) {
    s.push_back((uint32_t(args.size() + 1) << 16) | op);
    s.insert(s.end(), args.begin(), args.end());
  }
  static std::vector<uint32_t> str(const char* text) {
    std::vector<uint32_t> w((std::strlen(text) + 4) / 4, 0);
    std::memcpy(w.data(), text, std::strlen(text));
    return w;
  }

  std::vector<uint32_t> header_, imports_, entry_, exec_, annot_, types_, func_;

  // ---- types (deduplicated) ----
  uint32_t t_void() { return cached(k_void_, [&](uint32_t i) { emit(types_, OpTypeVoid, {i}); }); }
  uint32_t t_bool() { return cached(k_bool_, [&](uint32_t i) { emit(types_, OpTypeBool, {i}); }); }
  uint32_t t_u32() { return cached(k_u32_, [&](uint32_t i) { emit(types_, OpTypeInt, {i, 32, 0}); }); }
  uint32_t t_s32() { return cached(k_s32_, [&](uint32_t i) { emit(types_, OpTypeInt, {i, 32, 1}); }); }
  uint32_t t_float() { return cached(k_float_, [&](uint32_t i) { emit(types_, OpTypeFloat, {i, 32}); }); }
  uint32_t t_vec(uint32_t n) {
    const uint32_t f = t_float();
    return cached(k_vec_[n], [&](uint32_t i) { emit(types_, OpTypeVector, {i, f, n}); });
  }
  uint32_t t_ivec4() {
    const uint32_t s = t_s32();
    return cached(k_ivec4_, [&](uint32_t i) { emit(types_, OpTypeVector, {i, s, 4}); });
  }
  uint32_t t_bvec4() {
    const uint32_t b = t_bool();
    return cached(k_bvec4_, [&](uint32_t i) { emit(types_, OpTypeVector, {i, b, 4}); });
  }
  uint32_t t_fn_void() {
    const uint32_t v = t_void();
    return cached(k_fn_, [&](uint32_t i) { emit(types_, OpTypeFunction, {i, v}); });
  }
  uint32_t t_sampled_image() {
    const uint32_t f = t_float();
    const uint32_t img = cached(k_img_, [&](uint32_t i) { emit(types_, OpTypeImage, {i, f, 1, 0, 0, 0, 1, 0}); });
    return cached(k_simg_, [&](uint32_t i) { emit(types_, OpTypeSampledImage, {i, img}); });
  }
  uint32_t t_sampled_image_3d() {
    const uint32_t f = t_float();
    const uint32_t img = cached(k_img3d_, [&](uint32_t i) { emit(types_, OpTypeImage, {i, f, 2, 0, 0, 0, 1, 0}); });
    return cached(k_simg3d_, [&](uint32_t i) { emit(types_, OpTypeSampledImage, {i, img}); });
  }
  uint32_t t_sampled_image_shadow() {
    const uint32_t f = t_float();
    const uint32_t img = cached(k_img_sh_, [&](uint32_t i) { emit(types_, OpTypeImage, {i, f, 1, 1, 0, 0, 1, 0}); });
    return cached(k_simg_sh_, [&](uint32_t i) { emit(types_, OpTypeSampledImage, {i, img}); });
  }
  uint32_t t_ptr(uint32_t storage, uint32_t type) {
    auto key = std::make_pair(storage, type);
    auto it = ptrs_.find(key);
    if (it != ptrs_.end()) return it->second;
    const uint32_t i = id();
    emit(types_, OpTypePointer, {i, storage, type});
    ptrs_[key] = i;
    return i;
  }

  // ---- constants (deduplicated) ----
  uint32_t c_u32(uint32_t v) {
    auto it = u32s_.find(v);
    if (it != u32s_.end()) return it->second;
    const uint32_t t = t_u32(), i = id();
    emit(types_, OpConstant, {t, i, v});
    return u32s_[v] = i;
  }
  uint32_t c_s32(int32_t v) {
    auto it = s32s_.find(v);
    if (it != s32s_.end()) return it->second;
    const uint32_t t = t_s32(), i = id();
    emit(types_, OpConstant, {t, i, uint32_t(v)});
    return s32s_[v] = i;
  }
  uint32_t c_ivec4_zero() {
    return cached(k_ivec4_zero_, [&](uint32_t i) {
      const uint32_t z = c_s32(0), t = t_ivec4();
      emit(types_, OpConstantComposite, {t, i, z, z, z, z});
    });
  }
  uint32_t c_float(float f) {
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    auto it = floats_.find(bits);
    if (it != floats_.end()) return it->second;
    const uint32_t t = t_float(), i = id();
    emit(types_, OpConstant, {t, i, bits});
    floats_[bits] = i;
    return i;
  }
  uint32_t c_vec4(float a, float b, float c, float d) {
    const auto key = std::make_tuple(std::bit_cast<uint32_t>(a), std::bit_cast<uint32_t>(b), std::bit_cast<uint32_t>(c),
                                     std::bit_cast<uint32_t>(d));
    auto it = vec4s_.find(key);
    if (it != vec4s_.end()) return it->second;
    const uint32_t ca = c_float(a), cb = c_float(b), cc = c_float(c), cd = c_float(d);
    const uint32_t t = t_vec(4), i = id();
    emit(types_, OpConstantComposite, {t, i, ca, cb, cc, cd});
    vec4s_[key] = i;
    return i;
  }

  uint32_t c_bool(bool b) {
    return cached(b ? k_true_ : k_false_, [&](uint32_t i) { emit(types_, b ? OpConstantTrue : OpConstantFalse, {t_bool(), i}); });
  }
  uint32_t c_bvec4(bool x, bool y, bool z, bool w) {
    const uint32_t key = (uint32_t(x) << 0) | (uint32_t(y) << 1) | (uint32_t(z) << 2) | (uint32_t(w) << 3);
    auto it = bvec4s_.find(key);
    if (it != bvec4s_.end()) return it->second;
    const uint32_t bt = t_bvec4(), i = id();
    emit(types_, OpConstantComposite, {bt, i, c_bool(x), c_bool(y), c_bool(z), c_bool(w)});
    return bvec4s_[key] = i;
  }

  uint32_t global_var(uint32_t storage, uint32_t type) {
    const uint32_t p = t_ptr(storage, type), i = id();
    emit(types_, OpVariable, {p, i, storage});
    return i;
  }
  void decorate(uint32_t target, uint32_t dec, uint32_t value) { emit(annot_, OpDecorate, {target, dec, value}); }
  void decorate_flag(uint32_t target, uint32_t dec) { emit(annot_, OpDecorate, {target, dec}); }
  void member_decorate(uint32_t target, uint32_t member, uint32_t dec, uint32_t value) {
    emit(annot_, OpMemberDecorate, {target, member, dec, value});
  }

  std::vector<uint32_t> finish() {
    std::vector<uint32_t> out = {0x07230203u, 0x00010000u, 0, next_, 0};
    for (auto* s : {&header_, &imports_, &entry_, &exec_, &annot_, &types_, &func_}) out.insert(out.end(), s->begin(), s->end());
    return out;
  }

 private:
  template <class F>
  uint32_t cached(uint32_t& slot, F make) {
    if (!slot) {
      slot = id();
      make(slot);
    }
    return slot;
  }
  uint32_t next_ = 1;
  uint32_t k_void_ = 0, k_bool_ = 0, k_u32_ = 0, k_s32_ = 0, k_float_ = 0, k_vec_[5] = {}, k_ivec4_ = 0,
           k_ivec4_zero_ = 0, k_bvec4_ = 0, k_fn_ = 0, k_img_ = 0, k_simg_ = 0, k_img3d_ = 0, k_simg3d_ = 0,
           k_img_sh_ = 0, k_simg_sh_ = 0, k_true_ = 0, k_false_ = 0;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> ptrs_;
  std::map<uint32_t, uint32_t> u32s_;
  std::map<int32_t, uint32_t> s32s_;
  std::map<uint32_t, uint32_t> floats_;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, uint32_t> vec4s_;
  std::map<uint32_t, uint32_t> bvec4s_;
};

uint32_t fragment_input_location(uint32_t input) {
  // COL0 -> 0, COL1 -> 1, FOG -> 2, TEX0..TEX9 -> 3..12
  return input - 1;
}

}  // namespace

std::vector<uint32_t> spirv_passthrough_vs() {
  Module m;
  Module::emit(m.header_, OpCapability, {1});
  Module::emit(m.header_, OpMemoryModel, {0, 1});
  const uint32_t vec4 = m.t_vec(4);
  std::vector<uint32_t> in(14), out(13);
  for (uint32_t k = 0; k < 14; ++k) {
    in[k] = m.global_var(SC_Input, vec4);
    m.decorate(in[k], Dec_Location, k);
  }
  const uint32_t pos = m.global_var(SC_Output, vec4);
  m.decorate(pos, Dec_BuiltIn, BI_Position);
  for (uint32_t k = 0; k < 13; ++k) {
    out[k] = m.global_var(SC_Output, vec4);
    m.decorate(out[k], Dec_Location, k);
  }
  const uint32_t fnt = m.t_fn_void(), vt = m.t_void();
  const uint32_t main = m.id(), label = m.id();
  std::vector<uint32_t> ep = {0 /*Vertex*/, main};
  for (uint32_t w : Module::str("main")) ep.push_back(w);
  ep.insert(ep.end(), in.begin(), in.end());
  ep.push_back(pos);
  ep.insert(ep.end(), out.begin(), out.end());
  Module::emit(m.entry_, OpEntryPoint, ep);

  auto& f = m.func_;
  Module::emit(f, OpFunction, {vt, main, 0, fnt});
  Module::emit(f, OpLabel, {label});
  uint32_t v = m.id();
  Module::emit(f, OpLoad, {vec4, v, in[0]});
  Module::emit(f, OpStore, {pos, v});
  for (uint32_t k = 0; k < 13; ++k) {
    v = m.id();
    Module::emit(f, OpLoad, {vec4, v, in[k + 1]});
    Module::emit(f, OpStore, {out[k], v});
  }
  Module::emit(f, OpReturn, {});
  Module::emit(f, OpFunctionEnd, {});
  return m.finish();
}

bool spirv_translate_vp(const VpState& vp, std::vector<uint32_t>& spirv, SpirvVpMeta& meta, std::string* error) {
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
  meta = SpirvVpMeta{};

  // Determine which source slots (0, 1, 2) an instruction actually reads.
  auto src_read_mask = [](const VpInsn& i) -> uint32_t {
    uint32_t m = 0;
    switch (i.vec_op) {
      case VV_NOP: case VV_SFL: case VV_STR: break;
      case VV_MOV: case VV_ARL: case VV_FRC: case VV_FLR: case VV_SSG: m |= 1u; break;
      case VV_ADD: m |= 1u | 4u; break;
      case VV_MAD: m |= 1u | 2u | 4u; break;
      default: m |= 1u | 2u; break;
    }
    if (i.sca_op != VS_NOP) m |= 4u;
    return m;
  };

  // ---- scan the program for registers, inputs, constants and unsupported ops ----
  std::vector<VpInsn> prog;
  std::set<uint32_t> tmps, outs;
  bool uses_areg[2] = {false, false};
  bool uses_cc[2] = {false, false};
  bool found_end = false;
  for (uint32_t pc = vp.start; pc < vp.program.size() && prog.size() < 512; ++pc) {
    const VpInsn& i = vp.program[pc];
    switch (i.vec_op) {
      case VV_NOP: case VV_MOV: case VV_MUL: case VV_ADD: case VV_MAD: case VV_DP3: case VV_DPH: case VV_DP4:
      case VV_DST: case VV_MIN: case VV_MAX: case VV_SLT: case VV_SGE: case VV_ARL: case VV_FRC: case VV_FLR:
      case VV_SEQ: case VV_SFL: case VV_SGT: case VV_SLE: case VV_SNE: case VV_STR: case VV_SSG:
        break;
      default:
        return fail("vertex vector op " + std::to_string(i.vec_op) + " has no GPU translation");
    }
    switch (i.sca_op) {
      case VS_NOP: case VS_MOV: case VS_RCP: case VS_RCC: case VS_RSQ: case VS_EXP: case VS_LOG: case VS_LIT:
      case VS_LG2: case VS_EX2: case VS_SIN: case VS_COS:
        break;
      default:
        return fail("vertex scalar op " + std::to_string(i.sca_op) + " has no GPU translation");
    }
    const uint32_t rmask = src_read_mask(i);
    for (int k = 0; k < 3; ++k) {
      if (!(rmask & (1u << k))) continue;
      const VpSrc& s = i.src[k];
      if (s.type == VP_TEMP) tmps.insert(s.index & 0x3F);
      else if (s.type == VP_INPUT) meta.in_mask |= 1u << (i.input & 0xF);
      else if (s.type == VP_CONST) {
        if (i.index_const) {
          meta.uses_rel_const = true;
          uses_areg[i.addr_reg & 1] = true;
          meta.max_const = 511;
        } else {
          meta.max_const = std::max<uint32_t>(meta.max_const, i.constant & 511u);
        }
      }
    }
    if (i.vec_op == VV_ARL) uses_areg[i.addr_reg & 1] = true;
    else if (i.vec_op != VV_NOP) {
      if (i.dst_tmp != 0x3F) tmps.insert(i.dst_tmp & 0x3F);
      if (i.vec_result && i.dst_out < VPO_COUNT) outs.insert(i.dst_out);
    }
    if (i.sca_op != VS_NOP) {
      if (i.sca_dst_tmp != 0x3F) tmps.insert(i.sca_dst_tmp & 0x3F);
      if (!i.vec_result && i.dst_out < VPO_COUNT) outs.insert(i.dst_out);
    }
    if (i.cond_test || i.cond_update) uses_cc[i.cond_reg & 1] = true;
    prog.push_back(i);
    if (i.end) { found_end = true; break; }
  }
  if (!found_end) return fail("vertex program missing end marker");

  for (uint32_t k = 0; k < 16; ++k)
    if (meta.in_mask & (1u << k)) meta.in_loc[k] = uint8_t(meta.in_count++);

  Module m;
  Module::emit(m.header_, OpCapability, {1});
  const uint32_t glsl = m.id();
  {
    std::vector<uint32_t> a = {glsl};
    for (uint32_t w : Module::str("GLSL.std.450")) a.push_back(w);
    Module::emit(m.imports_, OpExtInstImport, a);
  }
  Module::emit(m.imports_, OpMemoryModel, {0, 1});

  const uint32_t tf = m.t_float(), ts32 = m.t_s32(), tv3 = m.t_vec(3), tv4 = m.t_vec(4), tivec4 = m.t_ivec4(),
                 tb = m.t_bool(), tbv4 = m.t_bvec4();
  const uint32_t f0 = m.c_float(0.0f), f1 = m.c_float(1.0f);
  const uint32_t v0 = m.c_vec4(0, 0, 0, 0), v1 = m.c_vec4(1, 1, 1, 1), vhpos0 = m.c_vec4(0, 0, 0, 1);

  // Interface variables: compacted vertex inputs, BuiltIn Position, and 13 varyings.
  std::vector<uint32_t> iface;
  uint32_t in_var[16] = {};
  for (uint32_t k = 0; k < 16; ++k) {
    if (!(meta.in_mask & (1u << k))) continue;
    in_var[k] = m.global_var(SC_Input, tv4);
    m.decorate(in_var[k], Dec_Location, meta.in_loc[k]);
    iface.push_back(in_var[k]);
  }
  const uint32_t pos_out = m.global_var(SC_Output, tv4);
  m.decorate(pos_out, Dec_BuiltIn, BI_Position);
  iface.push_back(pos_out);
  uint32_t vary_out[13] = {};
  for (uint32_t k = 0; k < 13; ++k) {
    vary_out[k] = m.global_var(SC_Output, tv4);
    m.decorate(vary_out[k], Dec_Location, k);
    iface.push_back(vary_out[k]);
  }

  // UBO at set 0, binding 17: { vec4 vp_scale; vec4 vp_offset; vec4 c[512]; }.
  const uint32_t t_arr = m.id();
  Module::emit(m.types_, OpTypeArray, {t_arr, tv4, m.c_u32(512)});
  m.decorate(t_arr, Dec_ArrayStride, 16);
  const uint32_t t_ubo = m.id();
  Module::emit(m.types_, OpTypeStruct, {t_ubo, tv4, tv4, t_arr});
  m.decorate_flag(t_ubo, Dec_Block);
  m.member_decorate(t_ubo, 0, Dec_Offset, 0);
  m.member_decorate(t_ubo, 1, Dec_Offset, 16);
  m.member_decorate(t_ubo, 2, Dec_Offset, 32);
  const uint32_t ubo_var = m.global_var(SC_Uniform, t_ubo);
  m.decorate(ubo_var, Dec_DescriptorSet, 0);
  m.decorate(ubo_var, Dec_Binding, 17);
  const uint32_t ptr_ubo_v4 = m.t_ptr(SC_Uniform, tv4);
  const uint32_t u0 = m.c_u32(0), u1 = m.c_u32(1), u2 = m.c_u32(2);

  const uint32_t fnt = m.t_fn_void(), tvoid = m.t_void();
  const uint32_t main = m.id(), label = m.id();
  {
    std::vector<uint32_t> ep = {0 /*Vertex*/, main};
    for (uint32_t w : Module::str("main")) ep.push_back(w);
    ep.insert(ep.end(), iface.begin(), iface.end());
    Module::emit(m.entry_, OpEntryPoint, ep);
  }

  std::vector<uint32_t> vars, body;
  const uint32_t ptr_fn_v4 = m.t_ptr(SC_Function, tv4);
  const uint32_t ptr_fn_iv4 = m.t_ptr(SC_Function, tivec4);
  std::map<uint32_t, uint32_t> tmp_var, out_var;
  for (uint32_t t : tmps) {
    tmp_var[t] = m.id();
    Module::emit(vars, OpVariable, {ptr_fn_v4, tmp_var[t], SC_Function, v0});
  }
  for (uint32_t o : outs) {
    out_var[o] = m.id();
    Module::emit(vars, OpVariable, {ptr_fn_v4, out_var[o], SC_Function, o == VPO_HPOS ? vhpos0 : v0});
  }
  uint32_t areg_var[2] = {}, cc_var[2] = {};
  for (int k = 0; k < 2; ++k) {
    if (uses_areg[k]) {
      areg_var[k] = m.id();
      Module::emit(vars, OpVariable, {ptr_fn_iv4, areg_var[k], SC_Function, m.c_ivec4_zero()});
    }
    if (uses_cc[k]) {
      cc_var[k] = m.id();
      Module::emit(vars, OpVariable, {ptr_fn_v4, cc_var[k], SC_Function, v0});
    }
  }

  auto op2 = [&](uint32_t op, uint32_t type, uint32_t a, uint32_t b) {
    const uint32_t r = m.id();
    Module::emit(body, op, {type, r, a, b});
    return r;
  };
  auto ext = [&](uint32_t inst, uint32_t type, std::initializer_list<uint32_t> args) {
    std::vector<uint32_t> a = {type, m.id(), glsl, inst};
    a.insert(a.end(), args.begin(), args.end());
    Module::emit(body, OpExtInst, a);
    return a[1];
  };
  auto shuffle = [&](uint32_t type, uint32_t a, uint32_t b, std::initializer_list<uint32_t> idx) {
    std::vector<uint32_t> args = {type, m.id(), a, b};
    args.insert(args.end(), idx.begin(), idx.end());
    Module::emit(body, OpVectorShuffle, args);
    return args[1];
  };
  auto extract = [&](uint32_t v, uint32_t c) {
    const uint32_t r = m.id();
    Module::emit(body, OpCompositeExtract, {tf, r, v, c});
    return r;
  };
  auto construct4 = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    const uint32_t r = m.id();
    Module::emit(body, OpCompositeConstruct, {tv4, r, a, b, c, d});
    return r;
  };
  auto splat = [&](uint32_t f) { return construct4(f, f, f, f); };
  auto load = [&](uint32_t type, uint32_t var) {
    const uint32_t r = m.id();
    Module::emit(body, OpLoad, {type, r, var});
    return r;
  };
  auto select_cmp = [&](uint32_t op, uint32_t a, uint32_t b) {
    const uint32_t c = op2(op, tbv4, a, b);
    const uint32_t r = m.id();
    Module::emit(body, OpSelect, {tv4, r, c, v1, v0});
    return r;
  };
  auto dot3 = [&](uint32_t a, uint32_t b) {
    return op2(OpDot, tf, shuffle(tv3, a, a, {0, 1, 2}), shuffle(tv3, b, b, {0, 1, 2}));
  };

  auto load_const = [&](const VpInsn& i) -> uint32_t {
    uint32_t idx_id = 0;
    if (i.index_const) {
      const uint32_t av = load(tivec4, areg_var[i.addr_reg & 1]);
      const uint32_t ac = m.id();
      Module::emit(body, OpCompositeExtract, {ts32, ac, av, uint32_t(i.addr_swz & 3)});
      const uint32_t sum = op2(OpIAdd, ts32, m.c_s32(int32_t(i.constant)), ac);
      idx_id = op2(OpBitwiseAnd, ts32, sum, m.c_s32(511));
    } else {
      idx_id = m.c_u32(i.constant & 511u);
    }
    const uint32_t ptr = m.id();
    Module::emit(body, OpAccessChain, {ptr_ubo_v4, ptr, ubo_var, u2, idx_id});
    return load(tv4, ptr);
  };

  auto fetch_src = [&](const VpInsn& i, int k) -> uint32_t {
    const VpSrc& s = i.src[k];
    uint32_t v = v0;
    if (s.type == VP_TEMP) v = load(tv4, tmp_var[s.index & 0x3F]);
    else if (s.type == VP_INPUT) v = in_var[i.input & 0xF] ? load(tv4, in_var[i.input & 0xF]) : v0;
    else if (s.type == VP_CONST) v = load_const(i);
    if (!(s.swz[0] == 0 && s.swz[1] == 1 && s.swz[2] == 2 && s.swz[3] == 3))
      v = shuffle(tv4, v, v, {s.swz[0], s.swz[1], s.swz[2], s.swz[3]});
    if (s.abs) v = ext(G_FAbs, tv4, {v});
    if (s.neg) {
      const uint32_t r = m.id();
      Module::emit(body, OpFNegate, {tv4, r, v});
      v = r;
    }
    return v;
  };

  auto eval_cond = [&](const VpInsn& i) -> uint32_t {
    if (i.cond == 0) return m.c_bvec4(false, false, false, false);
    if (i.cond == 7) return m.c_bvec4(true, true, true, true);
    uint32_t cv = load(tv4, cc_var[i.cond_reg & 1]);
    if (!(i.cond_swz[0] == 0 && i.cond_swz[1] == 1 && i.cond_swz[2] == 2 && i.cond_swz[3] == 3))
      cv = shuffle(tv4, cv, cv, {i.cond_swz[0], i.cond_swz[1], i.cond_swz[2], i.cond_swz[3]});
    static constexpr uint32_t kCmp[7] = {
        0, OpFOrdLessThan, OpFOrdEqual, OpFOrdLessThanEqual, OpFOrdGreaterThan, OpFUnordNotEqual, OpFOrdGreaterThanEqual,
    };
    return op2(kCmp[i.cond & 7], tbv4, cv, v0);
  };

  for (const VpInsn& i : prog) {
    const uint32_t rmask = src_read_mask(i);
    const uint32_t a = (rmask & 1u) ? fetch_src(i, 0) : v0;
    const uint32_t b = (rmask & 2u) ? fetch_src(i, 1) : v0;
    const uint32_t c = (rmask & 4u) ? fetch_src(i, 2) : v0;
    const bool has_cond = i.cond_test && i.cond != 7;
    const uint32_t cond_bv = has_cond ? eval_cond(i) : 0;

    auto store_v4 = [&](uint32_t var, uint32_t r, uint8_t mask) {
      if (!mask || (i.cond_test && i.cond == 0)) return;
      uint32_t val = r;
      if (has_cond) {
        uint32_t bv = cond_bv;
        if (mask != 0xF) {
          const uint32_t mbv = m.c_bvec4((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0, (mask & 8) != 0);
          bv = op2(OpLogicalAnd, tbv4, bv, mbv);
        }
        const uint32_t old = load(tv4, var);
        const uint32_t sel = m.id();
        Module::emit(body, OpSelect, {tv4, sel, bv, r, old});
        val = sel;
      } else if (mask != 0xF) {
        const uint32_t old = load(tv4, var);
        val = shuffle(tv4, old, r,
                      {(mask & 1) ? 4u : 0u, (mask & 2) ? 5u : 1u, (mask & 4) ? 6u : 2u, (mask & 8) ? 7u : 3u});
      }
      Module::emit(body, OpStore, {var, val});
    };

    if (i.vec_op == VV_ARL) {
      if (i.vec_mask && !(i.cond_test && i.cond == 0)) {
        const uint32_t fl = ext(G_Floor, tv4, {a});
        const uint32_t iv = m.id();
        Module::emit(body, OpConvertFToS, {tivec4, iv, fl});
        const uint32_t av = areg_var[i.addr_reg & 1];
        uint32_t val = iv;
        if (has_cond) {
          uint32_t bv = cond_bv;
          if (i.vec_mask != 0xF) {
            const uint32_t mbv = m.c_bvec4((i.vec_mask & 1) != 0, (i.vec_mask & 2) != 0, (i.vec_mask & 4) != 0,
                                           (i.vec_mask & 8) != 0);
            bv = op2(OpLogicalAnd, tbv4, bv, mbv);
          }
          const uint32_t old = load(tivec4, av);
          const uint32_t sel = m.id();
          Module::emit(body, OpSelect, {tivec4, sel, bv, iv, old});
          val = sel;
        } else if (i.vec_mask != 0xF) {
          const uint32_t old = load(tivec4, av);
          val = shuffle(tivec4, old, iv,
                        {(i.vec_mask & 1) ? 4u : 0u, (i.vec_mask & 2) ? 5u : 1u, (i.vec_mask & 4) ? 6u : 2u,
                         (i.vec_mask & 8) ? 7u : 3u});
        }
        Module::emit(body, OpStore, {av, val});
      }
    } else if (i.vec_op != VV_NOP) {
      uint32_t r = v0;
      switch (i.vec_op) {
        case VV_MOV: r = a; break;
        case VV_MUL: r = op2(OpFMul, tv4, a, b); break;
        case VV_ADD: r = op2(OpFAdd, tv4, a, c); break;  // ADD reads src0 + src2
        case VV_MAD: r = op2(OpFAdd, tv4, op2(OpFMul, tv4, a, b), c); break;
        case VV_DP3: r = splat(dot3(a, b)); break;
        case VV_DPH: r = splat(op2(OpFAdd, tf, dot3(a, b), extract(b, 3))); break;
        case VV_DP4: r = splat(op2(OpDot, tf, a, b)); break;
        case VV_DST: r = construct4(f1, op2(OpFMul, tf, extract(a, 1), extract(b, 1)), extract(a, 2), extract(b, 3)); break;
        case VV_MIN: r = ext(G_FMin, tv4, {a, b}); break;
        case VV_MAX: r = ext(G_FMax, tv4, {a, b}); break;
        case VV_SLT: r = select_cmp(OpFOrdLessThan, a, b); break;
        case VV_SGE: r = select_cmp(OpFOrdGreaterThanEqual, a, b); break;
        case VV_SEQ: r = select_cmp(OpFOrdEqual, a, b); break;
        case VV_SGT: r = select_cmp(OpFOrdGreaterThan, a, b); break;
        case VV_SLE: r = select_cmp(OpFOrdLessThanEqual, a, b); break;
        case VV_SNE: r = select_cmp(OpFOrdNotEqual, a, b); break;
        case VV_SFL: r = v0; break;
        case VV_STR: r = v1; break;
        case VV_FRC: r = ext(G_Fract, tv4, {a}); break;
        case VV_FLR: r = ext(G_Floor, tv4, {a}); break;
        case VV_SSG: r = ext(G_FSign, tv4, {a}); break;
        default: break;
      }
      if (i.saturate) r = ext(G_FClamp, tv4, {r, v0, v1});
      if (i.dst_tmp != 0x3F) store_v4(tmp_var[i.dst_tmp & 0x3F], r, i.vec_mask);
      if (i.vec_result && i.dst_out < VPO_COUNT) store_v4(out_var[i.dst_out], r, i.vec_mask);
      if (i.cond_update) store_v4(cc_var[i.cond_reg & 1], r, i.vec_mask);
    }

    if (i.sca_op != VS_NOP) {
      const uint32_t s = c;
      uint32_t r = v0;
      switch (i.sca_op) {
        case VS_MOV: r = s; break;
        case VS_RCP: case VS_RCC: {
          const uint32_t sx = extract(s, 0);
          const uint32_t nz = op2(OpFOrdNotEqual, tb, sx, f0);
          const uint32_t denom = m.id();
          Module::emit(body, OpSelect, {tf, denom, nz, sx, f1});
          const uint32_t q = op2(OpFDiv, tf, f1, denom);
          uint32_t rv = m.id();
          Module::emit(body, OpSelect, {tf, rv, nz, q, f0});
          if (i.sca_op == VS_RCC) rv = ext(G_FClamp, tf, {rv, m.c_float(5.42101e-20f), m.c_float(1.884467e19f)});
          r = splat(rv);
          break;
        }
        case VS_RSQ:
          r = splat(ext(G_InverseSqrt, tf, {ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(s, 0)}), m.c_float(1e-20f)})}));
          break;
        case VS_EXP: {
          const uint32_t x = ext(G_FClamp, tf, {extract(s, 0), m.c_float(-126.0f), m.c_float(126.0f)});
          r = construct4(ext(G_Exp2, tf, {ext(G_Floor, tf, {x})}), ext(G_Fract, tf, {x}), ext(G_Exp2, tf, {x}), f1);
          break;
        }
        case VS_LOG: {
          const uint32_t x = ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(s, 0)}), m.c_float(1e-38f)});
          const uint32_t l = ext(G_Log2, tf, {x});
          const uint32_t e = ext(G_Floor, tf, {l});
          r = construct4(e, op2(OpFDiv, tf, x, ext(G_Exp2, tf, {e})), l, f1);
          break;
        }
        case VS_LIT: {
          const uint32_t x = extract(s, 0), y = extract(s, 1), w = extract(s, 3);
          const uint32_t mx = ext(G_FMax, tf, {x, f0}), my = ext(G_FMax, tf, {y, m.c_float(1e-20f)});
          const uint32_t cw = ext(G_FClamp, tf, {w, m.c_float(-128.0f), m.c_float(128.0f)});
          const uint32_t p = ext(G_Pow, tf, {my, cw});
          const uint32_t pos = op2(OpFOrdGreaterThan, tb, x, f0);
          const uint32_t z = m.id();
          Module::emit(body, OpSelect, {tf, z, pos, p, f0});
          r = construct4(f1, mx, z, f1);
          break;
        }
        case VS_LG2:
          r = splat(ext(G_Log2, tf, {ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(s, 0)}), m.c_float(1e-38f)})}));
          break;
        case VS_EX2:
          r = splat(ext(G_Exp2, tf, {ext(G_FClamp, tf, {extract(s, 0), m.c_float(-126.0f), m.c_float(126.0f)})}));
          break;
        case VS_SIN: r = splat(ext(G_Sin, tf, {extract(s, 0)})); break;
        case VS_COS: r = splat(ext(G_Cos, tf, {extract(s, 0)})); break;
        default: break;
      }
      if (i.saturate) r = ext(G_FClamp, tv4, {r, v0, v1});
      if (i.sca_dst_tmp != 0x3F) store_v4(tmp_var[i.sca_dst_tmp & 0x3F], r, i.sca_mask);
      if (!i.vec_result && i.dst_out < VPO_COUNT) store_v4(out_var[i.dst_out], r, i.sca_mask);
      if (i.cond_update && i.vec_op == VV_NOP) store_v4(cc_var[i.cond_reg & 1], r, i.sca_mask);
    }
  }

  // Transform RSX clip HPOS into Vulkan clip space for the surface viewport [0..fw, 0..fh].
  const uint32_t hpos = out_var.count(VPO_HPOS) ? load(tv4, out_var[VPO_HPOS]) : vhpos0;
  const uint32_t p_vs = m.id(), p_vo = m.id();
  Module::emit(body, OpAccessChain, {ptr_ubo_v4, p_vs, ubo_var, u0});
  Module::emit(body, OpAccessChain, {ptr_ubo_v4, p_vo, ubo_var, u1});
  const uint32_t vp_s = load(tv4, p_vs), vp_o = load(tv4, p_vo);
  const uint32_t hw = extract(hpos, 3);
  const uint32_t gx = op2(OpFAdd, tf, op2(OpFMul, tf, extract(hpos, 0), extract(vp_s, 0)), op2(OpFMul, tf, hw, extract(vp_o, 0)));
  const uint32_t gy = op2(OpFAdd, tf, op2(OpFMul, tf, extract(hpos, 1), extract(vp_s, 1)), op2(OpFMul, tf, hw, extract(vp_o, 1)));
  const uint32_t gz = op2(OpFAdd, tf, op2(OpFMul, tf, extract(hpos, 2), extract(vp_s, 2)), op2(OpFMul, tf, hw, extract(vp_o, 2)));
  Module::emit(body, OpStore, {pos_out, construct4(gx, gy, gz, hw)});

  static constexpr uint8_t kVaryVpo[13] = {VPO_COL0, VPO_COL1, VPO_FOG,
                                           VPO_TEX0, VPO_TEX0 + 1, VPO_TEX0 + 2, VPO_TEX0 + 3, VPO_TEX0 + 4,
                                           VPO_TEX0 + 5, VPO_TEX0 + 6, VPO_TEX0 + 7, VPO_TEX0 + 8, VPO_TEX0 + 9};
  for (uint32_t k = 0; k < 13; ++k) {
    const uint32_t v = out_var.count(kVaryVpo[k]) ? load(tv4, out_var[kVaryVpo[k]]) : v0;
    Module::emit(body, OpStore, {vary_out[k], v});
  }
  Module::emit(body, OpReturn, {});

  auto& f = m.func_;
  Module::emit(f, OpFunction, {tvoid, main, 0, fnt});
  Module::emit(f, OpLabel, {label});
  f.insert(f.end(), vars.begin(), vars.end());
  f.insert(f.end(), body.begin(), body.end());
  Module::emit(f, OpFunctionEnd, {});
  spirv = m.finish();
  return true;
}

bool spirv_translate_fp(const std::vector<FpInsn>& prog, bool fp32_output, std::vector<uint32_t>& spirv,
                        std::string* error, bool alpha_test, uint32_t alpha_func, uint8_t alpha_ref,
                        uint16_t tex_3d_mask, uint16_t tex_shadow_mask) {
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };

  // ---- scan the program for registers, inputs, textures and inline constants ----
  std::set<uint32_t> regs_r = {0}, regs_h = {0}, inputs, units;
  bool h0_written = false, uses_cc = false;
  uint32_t const_count = 0;
  for (const FpInsn& i : prog) {
    if (i.has_constant) ++const_count;
    for (const FpSrc& s : i.src) {
      if (s.type == FP_TEMP) (s.fp16 ? regs_h : regs_r).insert(s.index);
      if (s.type == FP_INPUT) inputs.insert(i.input);
    }
    if (!i.no_dest) {
      (i.dst_fp16 ? regs_h : regs_r).insert(i.dst);
      if (i.dst_fp16 && i.dst == 0) h0_written = true;
    }
    if (i.set_cond || i.cond != 7) uses_cc = true;
    if (i.op == FP_TEX || i.op == FP_TXP || i.op == FP_TXL || i.op == FP_TXB || i.op == FP_TXD) units.insert(i.tex);
    if (i.end) break;
  }

  Module m;
  Module::emit(m.header_, OpCapability, {1});
  const uint32_t glsl = m.id();
  {
    std::vector<uint32_t> a = {glsl};
    for (uint32_t w : Module::str("GLSL.std.450")) a.push_back(w);
    Module::emit(m.imports_, OpExtInstImport, a);
  }
  Module::emit(m.imports_, OpMemoryModel, {0, 1});

  const uint32_t tf = m.t_float(), tv2 = m.t_vec(2), tv3 = m.t_vec(3), tv4 = m.t_vec(4), tb = m.t_bool(),
                 tbv4 = m.t_bvec4();
  const uint32_t f0 = m.c_float(0.0f), f1 = m.c_float(1.0f);
  const uint32_t v0 = m.c_vec4(0, 0, 0, 0), v1 = m.c_vec4(1, 1, 1, 1);

  // Interface variables.
  std::vector<uint32_t> iface;
  std::map<uint32_t, uint32_t> in_var;  // fp input index -> variable
  for (uint32_t in : inputs) {
    if (in == FPI_WPOS) {
      const uint32_t v = m.global_var(SC_Input, tv4);
      m.decorate(v, Dec_BuiltIn, BI_FragCoord);
      in_var[in] = v;
      iface.push_back(v);
    } else if (in >= FPI_COL0 && in < FPI_TEX0 + 10) {
      const uint32_t v = m.global_var(SC_Input, tv4);
      m.decorate(v, Dec_Location, fragment_input_location(in));
      in_var[in] = v;
      iface.push_back(v);
    }
  }
  const uint32_t out_var = m.global_var(SC_Output, tv4);
  m.decorate(out_var, Dec_Location, 0);
  iface.push_back(out_var);
  std::map<uint32_t, uint32_t> tex_var;
  for (uint32_t u : units) {
    const bool is_3d = (tex_3d_mask & (1u << u)) != 0;
    const bool is_shadow = !is_3d && ((tex_shadow_mask & (1u << u)) != 0);
    const uint32_t ty = is_3d ? m.t_sampled_image_3d() : is_shadow ? m.t_sampled_image_shadow() : m.t_sampled_image();
    const uint32_t v = m.global_var(SC_UniformConstant, ty);
    m.decorate(v, Dec_DescriptorSet, 0);
    m.decorate(v, Dec_Binding, u);
    tex_var[u] = v;
  }
  uint32_t ubo_var = 0, ptr_ubo_v4 = 0, u0 = 0;
  if (const_count > 0) {
    const uint32_t t_arr = m.id();
    Module::emit(m.types_, OpTypeArray, {t_arr, tv4, m.c_u32(const_count)});
    m.decorate(t_arr, Dec_ArrayStride, 16);
    const uint32_t t_ubo = m.id();
    Module::emit(m.types_, OpTypeStruct, {t_ubo, t_arr});
    m.decorate_flag(t_ubo, Dec_Block);
    m.member_decorate(t_ubo, 0, Dec_Offset, 0);
    ubo_var = m.global_var(SC_Uniform, t_ubo);
    m.decorate(ubo_var, Dec_DescriptorSet, 0);
    m.decorate(ubo_var, Dec_Binding, 16);
    ptr_ubo_v4 = m.t_ptr(SC_Uniform, tv4);
    u0 = m.c_u32(0);
  }

  const uint32_t fnt = m.t_fn_void(), tvoid = m.t_void();
  const uint32_t main = m.id(), label = m.id();
  {
    std::vector<uint32_t> ep = {4 /*Fragment*/, main};
    for (uint32_t w : Module::str("main")) ep.push_back(w);
    ep.insert(ep.end(), iface.begin(), iface.end());
    Module::emit(m.entry_, OpEntryPoint, ep);
    Module::emit(m.exec_, OpExecutionMode, {main, 7 /*OriginUpperLeft*/});
  }

  // Function-local register variables, zero-initialised.
  std::vector<uint32_t> vars, body;
  const uint32_t ptr_fn_v4 = m.t_ptr(SC_Function, tv4);
  std::map<uint32_t, uint32_t> R, H;
  for (uint32_t r : regs_r) { R[r] = m.id(); Module::emit(vars, OpVariable, {ptr_fn_v4, R[r], SC_Function, v0}); }
  for (uint32_t h : regs_h) { H[h] = m.id(); Module::emit(vars, OpVariable, {ptr_fn_v4, H[h], SC_Function, v0}); }
  uint32_t cc_var = 0;
  if (uses_cc) {
    cc_var = m.id();
    Module::emit(vars, OpVariable, {ptr_fn_v4, cc_var, SC_Function, v0});
  }

  // ---- instruction helpers ----
  auto op2 = [&](uint32_t op, uint32_t type, uint32_t a, uint32_t b) {
    const uint32_t r = m.id();
    Module::emit(body, op, {type, r, a, b});
    return r;
  };
  auto ext = [&](uint32_t inst, uint32_t type, std::initializer_list<uint32_t> args) {
    std::vector<uint32_t> a = {type, 0, glsl, inst};
    a[1] = m.id();
    a.insert(a.end(), args.begin(), args.end());
    Module::emit(body, OpExtInst, a);
    return a[1];
  };
  auto shuffle = [&](uint32_t type, uint32_t a, uint32_t b, std::initializer_list<uint32_t> idx) {
    std::vector<uint32_t> args = {type, m.id(), a, b};
    args.insert(args.end(), idx.begin(), idx.end());
    Module::emit(body, OpVectorShuffle, args);
    return args[1];
  };
  auto extract = [&](uint32_t v, uint32_t c) {
    const uint32_t r = m.id();
    Module::emit(body, OpCompositeExtract, {tf, r, v, c});
    return r;
  };
  auto construct4 = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    const uint32_t r = m.id();
    Module::emit(body, OpCompositeConstruct, {tv4, r, a, b, c, d});
    return r;
  };
  auto splat = [&](uint32_t f) { return construct4(f, f, f, f); };
  auto load = [&](uint32_t type, uint32_t var) {
    const uint32_t r = m.id();
    Module::emit(body, OpLoad, {type, r, var});
    return r;
  };
  auto select_cmp = [&](uint32_t op, uint32_t a, uint32_t b) {
    const uint32_t c = op2(op, tbv4, a, b);
    const uint32_t r = m.id();
    Module::emit(body, OpSelect, {tv4, r, c, v1, v0});
    return r;
  };
  auto dot3 = [&](uint32_t a, uint32_t b) {
    return op2(OpDot, tf, shuffle(tv3, a, a, {0, 1, 2}), shuffle(tv3, b, b, {0, 1, 2}));
  };

  auto src = [&](const FpInsn& i, int k, uint32_t cur_const) -> uint32_t {
    const FpSrc& s = i.src[k];
    uint32_t v = v0;
    if (s.type == FP_TEMP) v = load(tv4, s.fp16 ? H[s.index] : R[s.index]);
    else if (s.type == FP_INPUT) v = in_var.count(i.input) ? load(tv4, in_var[i.input]) : v0;
    else if (s.type == FP_CONST) v = cur_const;
    if (!(s.swz[0] == 0 && s.swz[1] == 1 && s.swz[2] == 2 && s.swz[3] == 3))
      v = shuffle(tv4, v, v, {s.swz[0], s.swz[1], s.swz[2], s.swz[3]});
    if (s.abs) v = ext(G_FAbs, tv4, {v});
    if (s.neg) {
      const uint32_t r = m.id();
      Module::emit(body, OpFNegate, {tv4, r, v});
      v = r;
    }
    return v;
  };

  auto sample = [&](const FpInsn& i, uint32_t coord_v4, int mode) -> uint32_t {
    if (!tex_var.count(i.tex)) return v0;
    const bool is_3d = (tex_3d_mask & (1u << i.tex)) != 0;
    const bool is_shadow = !is_3d && ((tex_shadow_mask & (1u << i.tex)) != 0);
    auto safe_inv_w = [&] {
      const uint32_t w = extract(coord_v4, 3);
      const uint32_t is0 = op2(OpFOrdEqual, tb, w, f0);
      const uint32_t ww = m.id();
      Module::emit(body, OpSelect, {tf, ww, is0, f1, w});
      return op2(OpFDiv, tf, f1, ww);
    };
    if (is_shadow) {
      const uint32_t img = load(m.t_sampled_image_shadow(), tex_var[i.tex]);
      uint32_t c3 = shuffle(tv3, coord_v4, coord_v4, {0, 1, 2});
      if (mode == 1) c3 = op2(OpVectorTimesScalar, tv3, c3, safe_inv_w());
      const uint32_t coord = shuffle(tv2, c3, c3, {0, 1});
      const uint32_t dref = ext(G_FClamp, tf, {extract(c3, 2), f0, f1});
      const uint32_t d = m.id();
      if (mode == 2) Module::emit(body, OpImageSampleDrefExplicitLod, {tf, d, img, coord, dref, 0x2, extract(coord_v4, 3)});
      else if (mode == 3) Module::emit(body, OpImageSampleDrefImplicitLod, {tf, d, img, coord, dref, 0x1, extract(coord_v4, 3)});
      else Module::emit(body, OpImageSampleDrefImplicitLod, {tf, d, img, coord, dref});
      const uint32_t r = splat(d);
      if (i.exp_tex) {
        const uint32_t v2 = m.c_vec4(2.0f, 2.0f, 2.0f, 2.0f);
        return op2(OpFSub, tv4, op2(OpFMul, tv4, r, v2), v1);
      }
      return r;
    }
    const uint32_t img = load(is_3d ? m.t_sampled_image_3d() : m.t_sampled_image(), tex_var[i.tex]);
    const uint32_t cty = is_3d ? tv3 : tv2;
    uint32_t coord = is_3d ? shuffle(tv3, coord_v4, coord_v4, {0, 1, 2}) : shuffle(tv2, coord_v4, coord_v4, {0, 1});
    const uint32_t r = m.id();
    if (mode == 1) {  // TXP: divide by w (w == 0 treated as 1)
      coord = op2(OpVectorTimesScalar, cty, coord, safe_inv_w());
      Module::emit(body, OpImageSampleImplicitLod, {tv4, r, img, coord});
    } else if (mode == 2) {  // TXL: explicit lod in w
      Module::emit(body, OpImageSampleExplicitLod, {tv4, r, img, coord, 0x2, extract(coord_v4, 3)});
    } else if (mode == 3) {  // TXB: bias in w
      Module::emit(body, OpImageSampleImplicitLod, {tv4, r, img, coord, 0x1, extract(coord_v4, 3)});
    } else {
      Module::emit(body, OpImageSampleImplicitLod, {tv4, r, img, coord});
    }
    if (i.exp_tex) {
      const uint32_t v2 = m.c_vec4(2.0f, 2.0f, 2.0f, 2.0f);
      return op2(OpFSub, tv4, op2(OpFMul, tv4, r, v2), v1);
    }
    return r;
  };

  auto eval_cond = [&](uint8_t cond, const uint8_t swz[4]) -> uint32_t {
    if (cond == 0) return m.c_bvec4(false, false, false, false);
    if (cond == 7) return m.c_bvec4(true, true, true, true);
    uint32_t cv = load(tv4, cc_var);
    if (!(swz[0] == 0 && swz[1] == 1 && swz[2] == 2 && swz[3] == 3))
      cv = shuffle(tv4, cv, cv, {swz[0], swz[1], swz[2], swz[3]});
    static constexpr uint32_t kCmp[7] = {
        0, OpFOrdLessThan, OpFOrdEqual, OpFOrdLessThanEqual, OpFOrdGreaterThan, OpFUnordNotEqual, OpFOrdGreaterThanEqual,
    };
    return op2(kCmp[cond & 7], tbv4, cv, v0);
  };

  bool killed = false;
  uint32_t const_idx = 0;
  for (const FpInsn& i : prog) {
    uint32_t cur_const = v0;
    const uint32_t this_const_idx = const_idx;
    if (i.has_constant) ++const_idx;
    if (i.op == FP_NOP || i.op == FP_FENCT || i.op == FP_FENCB) { if (i.end) break; continue; }
    if (i.op == FP_KIL) {
      if (i.cond == 7) {
        killed = true;
        break;
      }
      if (i.cond != 0) {
        const uint32_t any = m.id();
        Module::emit(body, OpAny, {tb, any, eval_cond(i.cond, i.cond_swz)});
        const uint32_t kill_lbl = m.id(), merge_lbl = m.id();
        Module::emit(body, OpSelectionMerge, {merge_lbl, 0});
        Module::emit(body, OpBranchConditional, {any, kill_lbl, merge_lbl});
        Module::emit(body, OpLabel, {kill_lbl});
        Module::emit(body, OpKill, {});
        Module::emit(body, OpLabel, {merge_lbl});
      }
      if (i.end) break;
      continue;
    }
    if (i.has_constant) {
      const uint32_t ptr = m.id();
      Module::emit(body, OpAccessChain, {ptr_ubo_v4, ptr, ubo_var, u0, m.c_u32(this_const_idx)});
      cur_const = load(tv4, ptr);
    }
    const uint32_t a = src(i, 0, cur_const), b = src(i, 1, cur_const), c = src(i, 2, cur_const);
    uint32_t r = 0;
    switch (i.op) {
      case FP_MOV: r = a; break;
      case FP_MUL: r = op2(OpFMul, tv4, a, b); break;
      case FP_ADD: r = op2(OpFAdd, tv4, a, b); break;
      case FP_MAD: r = op2(OpFAdd, tv4, op2(OpFMul, tv4, a, b), c); break;
      case FP_DP3: r = splat(dot3(a, b)); break;
      case FP_DP4: r = splat(op2(OpDot, tf, a, b)); break;
      case FP_DP2: r = splat(op2(OpDot, tf, shuffle(tv2, a, a, {0, 1}), shuffle(tv2, b, b, {0, 1}))); break;
      case FP_DP2A:
        r = splat(op2(OpFAdd, tf, op2(OpDot, tf, shuffle(tv2, a, a, {0, 1}), shuffle(tv2, b, b, {0, 1})), extract(c, 0)));
        break;
      case FP_DST: r = construct4(f1, op2(OpFMul, tf, extract(a, 1), extract(b, 1)), extract(a, 2), extract(b, 3)); break;
      case FP_MIN: r = ext(G_FMin, tv4, {a, b}); break;
      case FP_MAX: r = ext(G_FMax, tv4, {a, b}); break;
      case FP_SLT: r = select_cmp(OpFOrdLessThan, a, b); break;
      case FP_SGE: r = select_cmp(OpFOrdGreaterThanEqual, a, b); break;
      case FP_SLE: r = select_cmp(OpFOrdLessThanEqual, a, b); break;
      case FP_SGT: r = select_cmp(OpFOrdGreaterThan, a, b); break;
      case FP_SNE: r = select_cmp(OpFOrdNotEqual, a, b); break;
      case FP_SEQ: r = select_cmp(OpFOrdEqual, a, b); break;
      case FP_SFL: r = v0; break;
      case FP_STR: r = v1; break;
      case FP_FRC: r = ext(G_Fract, tv4, {a}); break;
      case FP_FLR: r = ext(G_Floor, tv4, {a}); break;
      case FP_DDX: case FP_DDY: r = v0; break;  // matches the reference renderer
      case FP_TEX: case FP_TXD: r = sample(i, a, 0); break;
      case FP_TXP: r = sample(i, a, 1); break;
      case FP_TXL: r = sample(i, a, 2); break;
      case FP_TXB: r = sample(i, a, 3); break;
      case FP_RCP: {
        const uint32_t ax = extract(a, 0);
        const uint32_t nz = op2(OpFOrdNotEqual, tb, ax, f0);
        const uint32_t d = m.id();
        Module::emit(body, OpSelect, {tf, d, nz, ax, m.c_float(1e-20f)});
        r = splat(op2(OpFDiv, tf, f1, d));
        break;
      }
      case FP_RSQ:
        r = splat(ext(G_InverseSqrt, tf, {ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(a, 0)}), m.c_float(1e-20f)})}));
        break;
      case FP_EX2:
        r = splat(ext(G_Exp2, tf, {ext(G_FClamp, tf, {extract(a, 0), m.c_float(-126.0f), m.c_float(126.0f)})}));
        break;
      case FP_LG2:
        r = splat(ext(G_Log2, tf, {ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(a, 0)}), m.c_float(1e-38f)})}));
        break;
      case FP_SIN: r = splat(ext(G_Sin, tf, {extract(a, 0)})); break;
      case FP_COS: r = splat(ext(G_Cos, tf, {extract(a, 0)})); break;
      case FP_POW: {
        const uint32_t bx = ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(a, 0)}), m.c_float(1e-20f)});
        const uint32_t ex = ext(G_FClamp, tf, {extract(b, 0), m.c_float(-126.0f), m.c_float(126.0f)});
        r = splat(ext(G_Pow, tf, {bx, ex}));
        break;
      }
      case FP_LRP: r = ext(G_FMix, tv4, {c, b, a}); break;
      case FP_LIT: {
        const uint32_t x = extract(a, 0), y = extract(a, 1), w = extract(a, 3);
        const uint32_t mx = ext(G_FMax, tf, {x, f0}), my = ext(G_FMax, tf, {y, m.c_float(1e-20f)});
        const uint32_t cw = ext(G_FClamp, tf, {w, m.c_float(-128.0f), m.c_float(128.0f)});
        const uint32_t p = ext(G_Pow, tf, {my, cw});
        const uint32_t pos = op2(OpFOrdGreaterThan, tb, x, f0);
        const uint32_t z = m.id();
        Module::emit(body, OpSelect, {tf, z, pos, p, f0});
        r = construct4(f1, mx, z, f1);
        break;
      }
      case FP_NRM: {
        const uint32_t v3 = shuffle(tv3, a, a, {0, 1, 2});
        const uint32_t len2 = ext(G_FMax, tf, {dot3(a, a), m.c_float(1e-20f)});
        const uint32_t inv = ext(G_InverseSqrt, tf, {len2});
        const uint32_t n = op2(OpVectorTimesScalar, tv3, v3, inv);
        r = shuffle(tv4, n, a, {0, 1, 2, 6});
        break;
      }
      case FP_DIV: {
        const uint32_t bx = extract(b, 0);
        const uint32_t nz = op2(OpFOrdNotEqual, tb, bx, f0);
        const uint32_t d = m.id();
        Module::emit(body, OpSelect, {tf, d, nz, bx, m.c_float(1e-20f)});
        r = op2(OpFDiv, tv4, a, splat(d));
        break;
      }
      case FP_DIVSQ: {
        const uint32_t inv =
            ext(G_InverseSqrt, tf, {ext(G_FMax, tf, {ext(G_FAbs, tf, {extract(b, 0)}), m.c_float(1e-20f)})});
        r = op2(OpFMul, tv4, a, splat(inv));
        break;
      }
      case FP_REFL: {
        const uint32_t num = op2(OpFMul, tf, m.c_float(2.0f), dot3(a, b));
        const uint32_t den = ext(G_FMax, tf, {dot3(a, a), m.c_float(1e-20f)});
        const uint32_t dd = op2(OpFDiv, tf, num, den);
        const uint32_t s = op2(OpFSub, tv4, op2(OpVectorTimesScalar, tv4, a, dd), b);
        r = shuffle(tv4, s, b, {0, 1, 2, 7});
        break;
      }
      default:
        return fail("fragment op " + std::to_string(i.op) + " has no GPU translation yet");
    }
    if ((!i.no_dest || i.set_cond) && i.cond != 0) {
      static const float kScale[8] = {1, 2, 4, 8, 1, 0.5f, 0.25f, 0.125f};
      if (i.scale && i.scale != 4) r = op2(OpFMul, tv4, r, m.c_vec4(kScale[i.scale], kScale[i.scale], kScale[i.scale], kScale[i.scale]));
      if (i.saturate) r = ext(G_FClamp, tv4, {r, v0, v1});
      uint32_t cond_bv = 0;
      if (i.cond != 7) {
        cond_bv = eval_cond(i.cond, i.cond_swz);
        if (i.mask != 0xF) {
          const uint32_t mbv = m.c_bvec4((i.mask & 1) != 0, (i.mask & 2) != 0, (i.mask & 4) != 0, (i.mask & 8) != 0);
          cond_bv = op2(OpLogicalAnd, tbv4, cond_bv, mbv);
        }
      }
      auto store_masked = [&](uint32_t var) {
        uint32_t val = r;
        if (i.cond != 7) {
          const uint32_t old = load(tv4, var);
          const uint32_t sel = m.id();
          Module::emit(body, OpSelect, {tv4, sel, cond_bv, r, old});
          val = sel;
        } else if (i.mask != 0xF) {
          const uint32_t old = load(tv4, var);
          val = shuffle(tv4, old, r, {(i.mask & 1) ? 4u : 0u, (i.mask & 2) ? 5u : 1u, (i.mask & 4) ? 6u : 2u, (i.mask & 8) ? 7u : 3u});
        }
        Module::emit(body, OpStore, {var, val});
      };
      if (!i.no_dest) store_masked(i.dst_fp16 ? H[i.dst] : R[i.dst]);
      if (i.set_cond) store_masked(cc_var);
    }
    if (i.end) break;
  }

  const uint32_t af = (alpha_func >= 0x200 && alpha_func <= 0x207) ? alpha_func : 0x207u;
  if (killed) {
    Module::emit(body, OpKill, {});
  } else if (alpha_test && af == 0x200) {
    Module::emit(body, OpKill, {});
  } else {
    const uint32_t res = load(tv4, (fp32_output || !h0_written) ? R[0] : H[0]);
    if (alpha_test && af != 0x207) {
      static constexpr uint32_t kAlphaCmp[7] = {
          0, OpFOrdLessThan, OpFOrdEqual, OpFOrdLessThanEqual, OpFOrdGreaterThan, OpFOrdNotEqual, OpFOrdGreaterThanEqual,
      };
      const uint32_t a = ext(G_FClamp, tf, {extract(res, 3), f0, f1});
      const uint32_t pass = op2(kAlphaCmp[af - 0x200], tb, a, m.c_float(float(alpha_ref) / 255.0f));
      const uint32_t kill_lbl = m.id(), merge_lbl = m.id();
      Module::emit(body, OpSelectionMerge, {merge_lbl, 0});
      Module::emit(body, OpBranchConditional, {pass, merge_lbl, kill_lbl});
      Module::emit(body, OpLabel, {kill_lbl});
      Module::emit(body, OpKill, {});
      Module::emit(body, OpLabel, {merge_lbl});
    }
    Module::emit(body, OpStore, {out_var, res});
    Module::emit(body, OpReturn, {});
  }

  auto& f = m.func_;
  Module::emit(f, OpFunction, {tvoid, main, 0, fnt});
  Module::emit(f, OpLabel, {label});
  f.insert(f.end(), vars.begin(), vars.end());
  f.insert(f.end(), body.begin(), body.end());
  Module::emit(f, OpFunctionEnd, {});
  spirv = m.finish();
  return true;
}

std::vector<uint32_t> spirv_convert_d24_to_argb_fs() {
  Module m;
  Module::emit(m.header_, OpCapability, {1});
  const uint32_t glsl = m.id();
  {
    std::vector<uint32_t> a = {glsl};
    for (uint32_t w : Module::str("GLSL.std.450")) a.push_back(w);
    Module::emit(m.imports_, OpExtInstImport, a);
  }
  Module::emit(m.imports_, OpMemoryModel, {0, 1});
  const uint32_t tf = m.t_float(), tv2 = m.t_vec(2), tv4 = m.t_vec(4), tvoid = m.t_void(), fnt = m.t_fn_void();
  const uint32_t uv_in = m.global_var(SC_Input, tv4);
  m.decorate(uv_in, Dec_Location, 3);  // TEX0
  const uint32_t out_var = m.global_var(SC_Output, tv4);
  m.decorate(out_var, Dec_Location, 0);
  const uint32_t tex = m.global_var(SC_UniformConstant, m.t_sampled_image());
  m.decorate(tex, Dec_DescriptorSet, 0);
  m.decorate(tex, Dec_Binding, 0);
  const uint32_t main = m.id(), lbl = m.id();
  {
    std::vector<uint32_t> ep = {4, main};
    for (uint32_t w : Module::str("main")) ep.push_back(w);
    ep.push_back(uv_in);
    ep.push_back(out_var);
    Module::emit(m.entry_, OpEntryPoint, ep);
    Module::emit(m.exec_, OpExecutionMode, {main, 7});
  }
  auto& b = m.func_;
  auto op2 = [&](uint32_t op, uint32_t ty, uint32_t x, uint32_t y) {
    uint32_t r = m.id(); Module::emit(b, op, {ty, r, x, y}); return r;
  };
  auto ext = [&](uint32_t inst, uint32_t ty, std::initializer_list<uint32_t> args) {
    std::vector<uint32_t> a = {ty, m.id(), glsl, inst};
    a.insert(a.end(), args.begin(), args.end());
    Module::emit(b, OpExtInst, a);
    return a[1];
  };
  Module::emit(b, OpFunction, {tvoid, main, 0, fnt});
  Module::emit(b, OpLabel, {lbl});
  const uint32_t uv4 = m.id(); Module::emit(b, OpLoad, {tv4, uv4, uv_in});
  const uint32_t uv2 = m.id(); Module::emit(b, OpVectorShuffle, {tv2, uv2, uv4, uv4, 0, 1});
  const uint32_t img = m.id(); Module::emit(b, OpLoad, {m.t_sampled_image(), img, tex});
  const uint32_t samp = m.id(); Module::emit(b, OpImageSampleImplicitLod, {tv4, samp, img, uv2});
  const uint32_t zraw = m.id(); Module::emit(b, OpCompositeExtract, {tf, zraw, samp, 0});
  const uint32_t z = ext(G_FClamp, tf, {zraw, m.c_float(0.0f), m.c_float(1.0f)});
  const uint32_t d = ext(G_Floor, tf, {op2(OpFAdd, tf, op2(OpFMul, tf, z, m.c_float(16777215.0f)), m.c_float(0.5f))});
  const uint32_t d_hi = ext(G_Floor, tf, {op2(OpFMul, tf, d, m.c_float(1.0f / 65536.0f))});
  const uint32_t rem1 = op2(OpFSub, tf, d, op2(OpFMul, tf, d_hi, m.c_float(65536.0f)));
  const uint32_t d_mid = ext(G_Floor, tf, {op2(OpFMul, tf, rem1, m.c_float(1.0f / 256.0f))});
  const uint32_t d_lo = op2(OpFSub, tf, rem1, op2(OpFMul, tf, d_mid, m.c_float(256.0f)));
  const uint32_t inv255 = m.c_float(1.0f / 255.0f);
  // Matches write_back (D24S8 big-endian [d_hi, d_mid, d_lo, 0]) read as A8R8G8B8 (R=d_mid, G=d_lo, B=0, A=d_hi).
  const uint32_t res = m.id();
  Module::emit(b, OpCompositeConstruct,
               {tv4, res, op2(OpFMul, tf, d_mid, inv255), op2(OpFMul, tf, d_lo, inv255), m.c_float(0.0f),
                op2(OpFMul, tf, d_hi, inv255)});
  Module::emit(b, OpStore, {out_var, res});
  Module::emit(b, OpReturn, {});
  Module::emit(b, OpFunctionEnd, {});
  return m.finish();
}

std::vector<uint32_t> spirv_convert_abgr_to_y16x16_fs() {
  Module m;
  Module::emit(m.header_, OpCapability, {1});
  Module::emit(m.imports_, OpMemoryModel, {0, 1});
  const uint32_t tf = m.t_float(), tv2 = m.t_vec(2), tv4 = m.t_vec(4), tvoid = m.t_void(), fnt = m.t_fn_void();
  const uint32_t uv_in = m.global_var(SC_Input, tv4);
  m.decorate(uv_in, Dec_Location, 3);  // TEX0
  const uint32_t out_var = m.global_var(SC_Output, tv4);
  m.decorate(out_var, Dec_Location, 0);
  const uint32_t tex = m.global_var(SC_UniformConstant, m.t_sampled_image());
  m.decorate(tex, Dec_DescriptorSet, 0);
  m.decorate(tex, Dec_Binding, 0);
  const uint32_t main = m.id(), lbl = m.id();
  {
    std::vector<uint32_t> ep = {4, main};
    for (uint32_t w : Module::str("main")) ep.push_back(w);
    ep.push_back(uv_in);
    ep.push_back(out_var);
    Module::emit(m.entry_, OpEntryPoint, ep);
    Module::emit(m.exec_, OpExecutionMode, {main, 7});
  }
  auto& b = m.func_;
  auto op2 = [&](uint32_t op, uint32_t ty, uint32_t x, uint32_t y) {
    uint32_t r = m.id(); Module::emit(b, op, {ty, r, x, y}); return r;
  };
  auto ext = [&](uint32_t v, uint32_t idx) {
    uint32_t r = m.id(); Module::emit(b, OpCompositeExtract, {tf, r, v, idx}); return r;
  };
  Module::emit(b, OpFunction, {tvoid, main, 0, fnt});
  Module::emit(b, OpLabel, {lbl});
  const uint32_t uv4 = m.id(); Module::emit(b, OpLoad, {tv4, uv4, uv_in});
  const uint32_t uv2 = m.id(); Module::emit(b, OpVectorShuffle, {tv2, uv2, uv4, uv4, 0, 1});
  const uint32_t img = m.id(); Module::emit(b, OpLoad, {m.t_sampled_image(), img, tex});
  const uint32_t c = m.id(); Module::emit(b, OpImageSampleImplicitLod, {tv4, c, img, uv2});
  // write_back (!argb): guest bytes [a, b, g, r]. texel(0x95): R = (a*256+b)/65535, G = (g*256+r)/65535.
  const uint32_t hi_w = m.c_float(65280.0f / 65535.0f), lo_w = m.c_float(255.0f / 65535.0f);
  const uint32_t y16 = op2(OpFAdd, tf, op2(OpFMul, tf, ext(c, 3), hi_w), op2(OpFMul, tf, ext(c, 2), lo_w));
  const uint32_t x16 = op2(OpFAdd, tf, op2(OpFMul, tf, ext(c, 1), hi_w), op2(OpFMul, tf, ext(c, 0), lo_w));
  const uint32_t res = m.id();
  Module::emit(b, OpCompositeConstruct, {tv4, res, y16, x16, m.c_float(0.0f), m.c_float(1.0f)});
  Module::emit(b, OpStore, {out_var, res});
  Module::emit(b, OpReturn, {});
  Module::emit(b, OpFunctionEnd, {});
  return m.finish();
}

}  // namespace rsx
