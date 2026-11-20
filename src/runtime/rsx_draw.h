#pragma once
// Reference (CPU) implementation of RSX draw calls. Renders straight into the
// render targets in guest memory, so flips, readbacks and CPU access all see the
// same pixels a real RSX would produce. It favours being easy to verify over
// speed; the GPU backend will reuse the same decoded state and programs.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "rsx_program.h"

namespace rsx {

// Method indices (byte offset / 4) for the 3D engine state used by draws.
namespace m {
constexpr uint32_t CONTEXT_DMA_COLOR_A = 0x65, CONTEXT_DMA_ZETA = 0x66;
constexpr uint32_t SURFACE_CLIP_H = 0x80, SURFACE_CLIP_V = 0x81, SURFACE_FORMAT = 0x82, SURFACE_PITCH_A = 0x83,
                   SURFACE_COLOR_AOFFSET = 0x84, SURFACE_ZETA_OFFSET = 0x85, SURFACE_COLOR_TARGET = 0x88,
                   SURFACE_PITCH_Z = 0x8B;
constexpr uint32_t ALPHA_TEST_ENABLE = 0xC1, ALPHA_FUNC = 0xC2, ALPHA_REF = 0xC3;
constexpr uint32_t BLEND_ENABLE = 0xC4, BLEND_SFACTOR = 0xC5, BLEND_DFACTOR = 0xC6, BLEND_COLOR = 0xC7,
                   BLEND_EQUATION = 0xC8, COLOR_MASK = 0xC9,
                   STENCIL_TEST_ENABLE = 0xCA, STENCIL_MASK = 0xCB, STENCIL_FUNC = 0xCC,
                   STENCIL_FUNC_REF = 0xCD, STENCIL_FUNC_MASK = 0xCE, STENCIL_OP_FAIL = 0xCF,
                   STENCIL_OP_ZFAIL = 0xD0, STENCIL_OP_ZPASS = 0xD1, TWO_SIDED_STENCIL_TEST_ENABLE = 0xD2,
                   BACK_STENCIL_MASK = 0xD3, BACK_STENCIL_FUNC = 0xD4, BACK_STENCIL_FUNC_REF = 0xD5,
                   BACK_STENCIL_FUNC_MASK = 0xD6, BACK_STENCIL_OP_FAIL = 0xD7,
                   POLY_OFFSET_FILL_ENABLE = 0xD8, BACK_STENCIL_OP_ZFAIL = 0xD8, BACK_STENCIL_OP_ZPASS = 0xD9,
                   BLEND_COLOR2 = 0xDF;
constexpr uint32_t SCISSOR_H = 0x230, SCISSOR_V = 0x231, SHADER_PROGRAM = 0x239;
constexpr uint32_t VIEWPORT_H = 0x280, VIEWPORT_V = 0x281, VIEWPORT_OFFSET = 0x288, VIEWPORT_SCALE = 0x28C;
constexpr uint32_t POLYGON_OFFSET_SCALE_FACTOR = 0x298, POLYGON_OFFSET_BIAS = 0x299;
constexpr uint32_t DEPTH_FUNC = 0x29B, DEPTH_MASK = 0x29C, DEPTH_TEST_ENABLE = 0x29D;
constexpr uint32_t TRANSFORM_PROGRAM = 0x2E0;  // 32-word window
constexpr uint32_t VERTEX_ARRAY_OFFSET = 0x5A0, VERTEX_ARRAY_FORMAT = 0x5D0;  // 16 each
constexpr uint32_t VERTEX_DATA_BASE_OFFSET = 0x5CE, VERTEX_DATA_BASE_INDEX = 0x5CF;
constexpr uint32_t BEGIN_END = 0x602, DRAW_ARRAYS = 0x605, INDEX_ARRAY_ADDRESS = 0x607, INDEX_ARRAY_DMA = 0x608,
                   DRAW_INDEX_ARRAY = 0x609;
constexpr uint32_t CULL_FACE = 0x60C, FRONT_FACE = 0x60D, CULL_FACE_ENABLE = 0x60F;
constexpr uint32_t TEXTURE_BASE = 0x680;  // 8 words per unit: offset, format, address, control0, control1, filter, rect, border
constexpr uint32_t TEXTURE_CONTROL3 = 0x610;  // method 0x1840 + 4*unit: pitch (bits 0-19), depth (bits 20-31)
constexpr uint32_t SHADER_CONTROL = 0x758, ZSTENCIL_CLEAR_VALUE = 0x763, COLOR_CLEAR_VALUE = 0x764,
                   CLEAR_SURFACE = 0x765;
constexpr uint32_t TRANSFORM_PROGRAM_LOAD = 0x7A7, TRANSFORM_PROGRAM_START = 0x7A8,
                   TRANSFORM_CONSTANT_LOAD = 0x7BF, TRANSFORM_CONSTANT = 0x7C0;  // 32-word window
}  // namespace m

struct DrawCall {
  uint32_t primitive = 0;
  bool indexed = false;
  std::vector<std::pair<uint32_t, uint32_t>> ranges;  // (first, count)
};

struct DrawEnv {
  uint8_t* mem = nullptr;                      // guest address space
  const uint32_t* regs = nullptr;              // method registers, indexed by method index
  const VpState* vp = nullptr;
  const vec4* const_attribs = nullptr;         // 16 constant vertex attributes (default {0,0,0,1})
  std::function<uint32_t(uint32_t offset, uint32_t location)> to_address;  // RSX offset -> guest address
};

struct DrawStats {
  uint64_t triangles = 0, pixels = 0;
};

// A bound texture unit, resolved to guest memory.
struct TexUnit {
  bool enabled = false;
  const uint8_t* base = nullptr;
  uint32_t address = 0;  // guest address (for caching / overlap checks)
  uint32_t format = 0, width = 1, height = 1, depth = 1, pitch = 0;
  uint32_t dimension = 2;  // 1=1D, 2=2D, 3=3D
  uint32_t mipmap = 1;     // RSX mipmap level count (1..13)
  bool cubemap = false;
  bool linear_layout = false, unnormalized = false;
  uint32_t wrap_s = 1, wrap_t = 1, wrap_r = 1;
  uint32_t gamma = 0, zfunc = 7, border_color = 0;
  uint32_t remap = 0x0000AAE4u;  // NV4097_SET_TEXTURE_CONTROL1 channel remap (default identity)
  uint8_t signed_mask = 0;       // bit0=R, bit1=G, bit2=B, bit3=A (from TEXTURE_FILTER bits 28..31)
  bool bilinear = false, mag_linear = false, min_linear = false;
  bool use_mip = false, mip_linear = false;
  float lod_bias = 0.0f;
  bool is_3d() const { return dimension == 3 && !cubemap; }
  bool is_cube() const { return cubemap; }
  bool is_depth_fmt() const { return format == 0x8F || format == 0x90 || format == 0x91 || format == 0x92; }
  bool is_shadow() const { return !is_3d() && !cubemap && is_depth_fmt() && zfunc != 7; }
  bool is_srgb() const { return (gamma & 7) != 0 && !is_depth_fmt(); }
  uint32_t cube_face_stride() const;
  uint32_t byte_size() const;
};
// Decodes a whole texture into RGBA8 (R in the low byte of each u32).
void decode_texture_rgba(const TexUnit& t, std::vector<uint32_t>& out);

// Number of varyings passed from vertex to pixel stage: COL0, COL1, FOG, TEX0..TEX9.
constexpr int kVaryings = 13;

// A vertex after vertex program, clipping and viewport transform.
struct PreparedVertex {
  float x = 0, y = 0, z = 0, inv_w = 1;  // window coordinates, window z, 1/w
  vec4 vary[kVaryings];                  // raw (not divided by w)
};

// Everything the pixel stage needs: resolved render target, state, programs and
// the visible triangles (3 vertices each, already clipped and culled).
struct PreparedDraw {
  bool argb = true;  // false = A8B8G8R8 family
  uint32_t color_base = 0, pitch = 0;
  uint32_t depth_fmt = 2, zbase = 0, zpitch = 0;
  int min_x = 0, max_x = 0, min_y = 0, max_y = 0;  // pixel bounds (clip + scissor)
  uint32_t surface_w = 0, surface_h = 0;
  bool depth_test = false, depth_write = false;
  uint32_t depth_func = 0x201;
  bool stencil_test = false, two_sided_stencil = false;
  uint8_t stencil_mask = 0xFF, stencil_ref = 0, stencil_func_mask = 0xFF;
  uint32_t stencil_func = 0x207, stencil_op_fail = 0x1E00, stencil_op_zfail = 0x1E00, stencil_op_zpass = 0x1E00;
  uint8_t back_stencil_mask = 0xFF, back_stencil_ref = 0, back_stencil_func_mask = 0xFF;
  uint32_t back_stencil_func = 0x207, back_stencil_op_fail = 0x1E00, back_stencil_op_zfail = 0x1E00, back_stencil_op_zpass = 0x1E00;
  bool blend = false;
  uint32_t sf_rgb = 1, sf_a = 1, df_rgb = 0, df_a = 0, eq_rgb = 0x8006, eq_a = 0x8006;
  uint32_t color_mask = 0x01010101;
  bool alpha_test = false;
  uint32_t alpha_func = 0x207;
  uint8_t alpha_ref = 0;
  uint32_t blend_color = 0;
  uint32_t raw_tris = 0, rej_nonfinite = 0, rej_clip = 0, rej_cull = 0;
  float first_clip[3][4]{};
  bool cull = false;
  uint32_t cull_face = 0x405;
  bool front_ccw = true;
  bool poly_offset = false;
  float poly_offset_scale = 0.0f, poly_offset_bias = 0.0f;
  float vp_scale[3]{}, vp_offset[3]{};
  uint32_t aa_x = 1, aa_y = 1;
  bool fp32_out = true;
  std::vector<FpInsn> fp;
  std::vector<vec4> fp_consts;  // inline constants extracted from fp (for UBO binding 16)
  uint64_t fp_hash = 0;
  TexUnit tex[16];
  std::vector<PreparedVertex> tris;
};

// Resolves render target, fixed-function state, textures and decoded fragment program
// without running the vertex program on the CPU (used by the GPU vertex-shader path
// and as the first half of prepare_draw).
bool prepare_draw_state(const DrawEnv& env, PreparedDraw& out, std::string* error);

// Packs unique vertex inputs (only the attributes set in `in_mask`, in ascending
// attribute-index order, 4 floats per attribute) into `dst_vtx`, and expanded
// triangle-list uint32_t indices into `dst_idx`.
bool pack_indexed_geometry(const DrawEnv& env, const DrawCall& call, uint32_t in_mask, uint32_t in_count,
                           float* dst_vtx, uint32_t max_vtx, uint32_t& out_vtx,
                           uint32_t* dst_idx, uint32_t max_idx, uint32_t& out_idx, std::string* error);

// Front half of a draw: vertex fetch, vertex program, primitive assembly, clipping,
// culling and state resolution. Returns false (with a reason) for unsupported draws.
bool prepare_draw(const DrawEnv& env, const DrawCall& call, PreparedDraw& out, std::string* error);

// Back half on the CPU: rasterise, run the fragment program, depth test, blend.
bool raster_cpu(const DrawEnv& env, const PreparedDraw& d, DrawStats& stats, std::string* error);

// prepare_draw + raster_cpu.
bool draw(const DrawEnv& env, const DrawCall& call, DrawStats& stats, std::string* error);

// Clears depth/stencil of the current zeta surface (CLEAR_SURFACE bits 0x1 / 0x2).
void clear_depth(const DrawEnv& env, uint32_t mask);

// Default register values that differ from zero after an RSX reset.
void reset_defaults(uint32_t* regs);

// Sample scale factors implied by NV4097_SET_SURFACE_FORMAT's antialias field
// (bits 12..15): 3 = DIAGONAL_CENTERED_2 (2x1), 4/5 = SQUARE_*_4 (2x2).
inline void surface_aa_scale(uint32_t surface_format, uint32_t& sx, uint32_t& sy) {
  switch ((surface_format >> 12) & 0xF) {
    case 3: sx = 2; sy = 1; break;
    case 4: case 5: sx = 2; sy = 2; break;
    default: sx = 1; sy = 1; break;
  }
}

}  // namespace rsx
