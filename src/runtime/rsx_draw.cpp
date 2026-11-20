#include "rsx_draw.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace rsx {
namespace {

// ---- guest memory access (big-endian) ---------------------------------------------
uint16_t be16(const uint8_t* p) { return uint16_t((p[0] << 8) | p[1]); }
uint32_t be32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
void put_be32(uint8_t* p, uint32_t v) {
  p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16); p[2] = uint8_t(v >> 8); p[3] = uint8_t(v);
}
void put_be16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v >> 8); p[1] = uint8_t(v); }
float fbits(uint32_t v) { return std::bit_cast<float>(v); }
float be_float(const uint8_t* p) { return fbits(be32(p)); }

// Bytes per texel of the uncompressed RSX texture formats.
uint32_t texel_bytes(uint32_t f) {
  switch (f) {
    case 0x81: return 1;                                    // B8
    case 0x85: case 0x8F: case 0x90: case 0x95: case 0x9C:  // A8R8G8B8, DEPTH24_D8(_FLOAT), Y16_X16, X32_FLOAT
    case 0x9E: case 0x9F: return 4;                         // D8R8G8B8, Y16_X16_FLOAT
    case 0x9A: return 8;                                    // W16_Z16_Y16_X16_FLOAT
    case 0x9B: return 16;                                   // W32_Z32_Y32_X32_FLOAT
    default: return 2;                                      // 16-bit formats
  }
}

float half_to_float(uint16_t h) {
  const uint32_t sign = uint32_t(h >> 15) << 31, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else {  // subnormal
      float f = std::ldexp(float(mant), -24);
      return (h >> 15) ? -f : f;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(bits);
}

// ---- vertex fetch -----------------------------------------------------------------------
enum VertexType { VT_S1 = 1, VT_F = 2, VT_SF = 3, VT_UB = 4, VT_S32K = 5, VT_CMP = 6, VT_UB256 = 7 };

struct Attrib {
  const uint8_t* base = nullptr;
  uint32_t type = 0, size = 0, stride = 0;
};

vec4 fetch_attrib(const Attrib& a, uint32_t index) {
  vec4 v{0, 0, 0, 1};
  if (!a.base || !a.size) return v;
  const uint8_t* p = a.base + size_t(index) * a.stride;
  switch (a.type) {
    case VT_F: for (uint32_t c = 0; c < a.size && c < 4; ++c) v[c] = fbits(be32(p + 4 * c)); break;
    case VT_SF: for (uint32_t c = 0; c < a.size && c < 4; ++c) v[c] = half_to_float(be16(p + 2 * c)); break;
    case VT_S1:
      for (uint32_t c = 0; c < a.size && c < 4; ++c) v[c] = std::max(-1.0f, float(int16_t(be16(p + 2 * c))) / 32767.0f);
      break;
    case VT_S32K: for (uint32_t c = 0; c < a.size && c < 4; ++c) v[c] = float(int16_t(be16(p + 2 * c))); break;
    case VT_UB: for (uint32_t c = 0; c < a.size && c < 4; ++c) v[c] = p[c] / 255.0f; break;
    case VT_UB256: for (uint32_t c = 0; c < a.size && c < 4; ++c) v[c] = float(p[c]); break;
    case VT_CMP: {  // x:11 y:11 z:10, signed normalised
      const uint32_t w = be32(p);
      auto sx = [](uint32_t bits, int n) { const int32_t s = int32_t(bits << (32 - n)) >> (32 - n); return float(s); };
      v[0] = sx(w & 0x7FF, 11) / 1023.0f;
      v[1] = sx((w >> 11) & 0x7FF, 11) / 1023.0f;
      v[2] = sx(w >> 22, 10) / 511.0f;
      break;
    }
    default: break;
  }
  return v;
}

// ---- textures ----------------------------------------------------------------------------
uint32_t morton(uint32_t x, uint32_t y, uint32_t lw, uint32_t lh) {
  uint32_t off = 0, shift = 0;
  const uint32_t both = std::min(lw, lh);
  for (uint32_t i = 0; i < both; ++i) {
    off |= ((x >> i) & 1) << shift++;
    off |= ((y >> i) & 1) << shift++;
  }
  off |= (lw > lh ? (x >> both) : (y >> both)) << shift;
  return off;
}

uint32_t morton3d(uint32_t x, uint32_t y, uint32_t z, uint32_t lw, uint32_t lh, uint32_t ld) {
  uint32_t off = 0, shift = 0;
  const uint32_t max_l = std::max({lw, lh, ld});
  for (uint32_t i = 0; i < max_l; ++i) {
    if (i < lw) off |= ((x >> i) & 1u) << shift++;
    if (i < lh) off |= ((y >> i) & 1u) << shift++;
    if (i < ld) off |= ((z >> i) & 1u) << shift++;
  }
  return off;
}

int wrap_coord(int c, int size, uint32_t mode) {
  switch (mode) {
    case 1: {  // wrap
      const int m = c % size;
      return m < 0 ? m + size : m;
    }
    case 2: {  // mirror
      const int period = 2 * size;
      int m = c % period;
      if (m < 0) m += period;
      return m < size ? m : period - 1 - m;
    }
    default:  // clamp variants (border treated as clamp)
      return std::clamp(c, 0, size - 1);
  }
}

vec4 unpack565(uint16_t v) {
  return {((v >> 11) & 31) / 31.0f, ((v >> 5) & 63) / 63.0f, (v & 31) / 31.0f, 1.0f};
}

// One DXT block (little-endian, as on PC). Returns the texel (px, py) inside the block.
vec4 dxt_texel(const uint8_t* blk, uint32_t fmt, uint32_t px, uint32_t py) {
  const uint8_t* color = fmt == 0x86 ? blk : blk + 8;
  const uint16_t c0 = uint16_t(color[0] | (color[1] << 8)), c1 = uint16_t(color[2] | (color[3] << 8));
  const uint32_t bits = uint32_t(color[4]) | (uint32_t(color[5]) << 8) | (uint32_t(color[6]) << 16) | (uint32_t(color[7]) << 24);
  const uint32_t sel = (bits >> (2 * (py * 4 + px))) & 3;
  const vec4 a = unpack565(c0), b = unpack565(c1);
  vec4 out;
  const bool four = fmt != 0x86 || c0 > c1;
  if (sel == 0) out = a;
  else if (sel == 1) out = b;
  else if (four) {
    const float t = sel == 2 ? 1.0f / 3 : 2.0f / 3;
    out = {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, 1.0f};
  } else if (sel == 2) {
    out = {(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, 1.0f};
  } else {
    out = {0, 0, 0, 0};  // DXT1 transparent
  }
  const uint32_t i = py * 4 + px;
  if (fmt == 0x87) {  // explicit 4-bit alpha
    out[3] = ((blk[i / 2] >> ((i & 1) * 4)) & 0xF) / 15.0f;
  } else if (fmt == 0x88) {  // interpolated alpha
    const float a0 = blk[0] / 255.0f, a1 = blk[1] / 255.0f;
    uint64_t abits = 0;
    for (int k = 0; k < 6; ++k) abits |= uint64_t(blk[2 + k]) << (8 * k);
    const uint32_t s = uint32_t(abits >> (3 * i)) & 7;
    float al;
    if (s == 0) al = a0;
    else if (s == 1) al = a1;
    else if (blk[0] > blk[1]) al = (a0 * (8 - s) + a1 * (s - 1)) / 7.0f;
    else if (s < 6) al = (a0 * (6 - s) + a1 * (s - 1)) / 5.0f;
    else al = s == 6 ? 0.0f : 1.0f;
    out[3] = al;
  }
  return out;
}

vec4 texel(const TexUnit& t, int x, int y, int z = 0) {
  const uint32_t ux = uint32_t(wrap_coord(x, int(t.width), t.wrap_s));
  const uint32_t uy = uint32_t(wrap_coord(y, int(t.height), t.wrap_t));
  const uint32_t uz = t.is_3d() ? uint32_t(wrap_coord(z, int(t.depth), t.wrap_r)) : 0u;
  const uint32_t f = t.format;
  if (f == 0x86 || f == 0x87 || f == 0x88) {
    const uint32_t bsize = f == 0x86 ? 8 : 16;
    const uint32_t bw = (t.width + 3) / 4, bh = (t.height + 3) / 4;
    const uint8_t* blk = t.base + ((size_t(uz) * bh + uy / 4) * bw + ux / 4) * bsize;
    return dxt_texel(blk, f, ux & 3, uy & 3);
  }
  const uint32_t bpp = texel_bytes(f);
  size_t off;
  if (t.linear_layout) {
    const size_t stride = t.pitch ? t.pitch : size_t(t.width) * bpp;
    off = (size_t(uz) * t.height + uy) * stride + size_t(ux) * bpp;
  } else if (t.is_3d()) {
    off = size_t(morton3d(ux, uy, uz, uint32_t(std::countr_zero(std::bit_ceil(t.width))),
                          uint32_t(std::countr_zero(std::bit_ceil(t.height))),
                          uint32_t(std::countr_zero(std::bit_ceil(t.depth))))) * bpp;
  } else {
    off = size_t(morton(ux, uy, uint32_t(std::countr_zero(std::bit_ceil(t.width))),
                        uint32_t(std::countr_zero(std::bit_ceil(t.height))))) * bpp;
  }
  const uint8_t* p = t.base + off;
  switch (f) {
    case 0x85: {  // A8R8G8B8
      const uint32_t v = be32(p);
      return {((v >> 16) & 0xFF) / 255.0f, ((v >> 8) & 0xFF) / 255.0f, (v & 0xFF) / 255.0f, (v >> 24) / 255.0f};
    }
    case 0x84: return unpack565(be16(p));
    case 0x82: {
      const uint16_t v = be16(p);
      return {((v >> 10) & 31) / 31.0f, ((v >> 5) & 31) / 31.0f, (v & 31) / 31.0f, float(v >> 15)};
    }
    case 0x83: {
      const uint16_t v = be16(p);
      return {((v >> 8) & 15) / 15.0f, ((v >> 4) & 15) / 15.0f, (v & 15) / 15.0f, (v >> 12) / 15.0f};
    }
    case 0x81: {
      const float b = p[0] / 255.0f;
      return {b, b, b, b};
    }
    case 0x8B: return {p[0] / 255.0f, p[1] / 255.0f, 0.0f, 1.0f};  // G8B8
    case 0x94: { const float v = be16(p) / 65535.0f; return {v, v, v, v}; }  // X16
    case 0x95: return {be16(p) / 65535.0f, be16(p + 2) / 65535.0f, 0.0f, 1.0f};  // Y16_X16
    case 0x97: {  // R5G5B5A1
      const uint16_t v = be16(p);
      return {(v >> 11) / 31.0f, ((v >> 6) & 31) / 31.0f, ((v >> 1) & 31) / 31.0f, float(v & 1)};
    }
    case 0x9D: {  // D1R5G5B5
      const uint16_t v = be16(p);
      return {((v >> 10) & 31) / 31.0f, ((v >> 5) & 31) / 31.0f, (v & 31) / 31.0f, 1.0f};
    }
    case 0x9E: {  // D8R8G8B8
      const uint32_t v = be32(p);
      return {((v >> 16) & 0xFF) / 255.0f, ((v >> 8) & 0xFF) / 255.0f, (v & 0xFF) / 255.0f, 1.0f};
    }
    case 0x9A: return {half_to_float(be16(p)), half_to_float(be16(p + 2)), half_to_float(be16(p + 4)), half_to_float(be16(p + 6))};
    case 0x9B: return {be_float(p), be_float(p + 4), be_float(p + 8), be_float(p + 12)};
    case 0x9C: { const float v = be_float(p); return {v, v, v, v}; }  // X32_FLOAT
    case 0x9F: return {half_to_float(be16(p)), half_to_float(be16(p + 2)), 0.0f, 1.0f};  // Y16_X16_FLOAT
    case 0x8F: case 0x90: {  // DEPTH24_D8 (fixed / float): depth in the top 24 bits
      const float z = (be32(p) >> 8) / 16777215.0f;
      return {z, z, z, z};
    }
    case 0x91: case 0x92: { const float z = be16(p) / 65535.0f; return {z, z, z, z}; }  // DEPTH16
    default: return {0, 0, 0, 1};
  }
}

struct PixelEnv {
  TexUnit tex[16];
};

float srgb_to_linear_c(float s) {
  return s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
}

float shadow_cmp(uint32_t zfunc, float d_tex, float d_ref) {
  switch (zfunc) {
    case 0: return 0.0f;
    case 1: return d_tex < d_ref ? 1.0f : 0.0f;
    case 2: return d_tex == d_ref ? 1.0f : 0.0f;
    case 3: return d_tex <= d_ref ? 1.0f : 0.0f;
    case 4: return d_tex > d_ref ? 1.0f : 0.0f;
    case 5: return d_tex != d_ref ? 1.0f : 0.0f;
    case 6: return d_tex >= d_ref ? 1.0f : 0.0f;
    default: return 1.0f;
  }
}

vec4 sample(void* user, uint32_t unit, const vec4& coord) {
  const TexUnit& t = static_cast<PixelEnv*>(user)->tex[unit & 15];
  if (!t.enabled || !t.base) return {0, 0, 0, 1};
  auto fetch = [&](int x, int y, int z) -> vec4 {
    vec4 c = texel(t, x, y, z);
    if (t.is_shadow()) {
      const float r = shadow_cmp(t.zfunc, c[0], std::clamp(coord[2], 0.0f, 1.0f));
      return {r, r, r, r};
    }
    if (t.signed_mask && t.format != 0x9A && t.format != 0x9B && t.format != 0x9C && t.format != 0x9F) {
      for (int k = 0; k < 4; ++k) {
        if (t.signed_mask & (1u << k)) {
          const uint8_t u = uint8_t(std::clamp(c[k], 0.0f, 1.0f) * 255.0f + 0.5f);
          c[k] = std::max(-1.0f, float(int8_t(u)) / 127.0f);
        }
      }
    }
    if (t.is_srgb()) {
      for (int k = 0; k < 3; ++k) c[k] = srgb_to_linear_c(std::clamp(c[k], 0.0f, 1.0f));
    }
    if ((t.remap & 0xFFFFu) != 0xAAE4u) {
      const float in_v[4] = {c[3], c[0], c[1], c[2]};  // 0=A, 1=R, 2=G, 3=B
      auto map_ch = [&](uint32_t out_op, uint32_t in_sel) -> float {
        if (out_op == 0) return 0.0f;
        if (out_op == 1) return 1.0f;
        return in_v[in_sel & 3];
      };
      c = {
          map_ch((t.remap >> 10) & 3, (t.remap >> 2) & 3),  // R
          map_ch((t.remap >> 12) & 3, (t.remap >> 4) & 3),  // G
          map_ch((t.remap >> 14) & 3, (t.remap >> 6) & 3),  // B
          map_ch((t.remap >> 8) & 3,  (t.remap >> 0) & 3),  // A
      };
    }
    return c;
  };
  float u = coord[0], v = coord[1], w = coord[2];
  if (!t.unnormalized) {
    u *= float(t.width);
    v *= float(t.height);
    if (t.is_3d()) w *= float(t.depth);
  }
  if (!t.bilinear) return fetch(int(std::floor(u)), int(std::floor(v)), t.is_3d() ? int(std::floor(w)) : 0);
  u -= 0.5f;
  v -= 0.5f;
  const int x0 = int(std::floor(u)), y0 = int(std::floor(v));
  const float fx = u - x0, fy = v - y0;
  auto bilerp = [&](int z) -> vec4 {
    const vec4 a = fetch(x0, y0, z), b = fetch(x0 + 1, y0, z), c = fetch(x0, y0 + 1, z), d = fetch(x0 + 1, y0 + 1, z);
    vec4 r;
    for (int k = 0; k < 4; ++k)
      r[k] = (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy;
    return r;
  };
  if (!t.is_3d()) return bilerp(0);
  w -= 0.5f;
  const int z0 = int(std::floor(w));
  const float fz = w - z0;
  const vec4 s0 = bilerp(z0), s1 = bilerp(z0 + 1);
  vec4 r;
  for (int k = 0; k < 4; ++k) r[k] = s0[k] * (1 - fz) + s1[k] * fz;
  return r;
}

// ---- blending -----------------------------------------------------------------------------
float factor(uint32_t f, const vec4& s, const vec4& d, int c) {
  switch (f) {
    case 0: return 0.0f;
    case 1: return 1.0f;
    case 0x300: return s[c];
    case 0x301: return 1 - s[c];
    case 0x302: return s[3];
    case 0x303: return 1 - s[3];
    case 0x304: return d[3];
    case 0x305: return 1 - d[3];
    case 0x306: return d[c];
    case 0x307: return 1 - d[c];
    case 0x308: return c == 3 ? 1.0f : std::min(s[3], 1 - d[3]);
    default: return 1.0f;
  }
}
float equation(uint32_t eq, float s, float d) {
  switch (eq) {
    case 0x8007: return std::min(s, d);
    case 0x8008: return std::max(s, d);
    case 0x800A: return s - d;
    case 0x800B: return d - s;
    default: return s + d;
  }
}

bool depth_pass(uint32_t func, uint32_t incoming, uint32_t stored) {
  switch (func) {
    case 0x200: return false;
    case 0x201: return incoming < stored;
    case 0x202: return incoming == stored;
    case 0x203: return incoming <= stored;
    case 0x204: return incoming > stored;
    case 0x205: return incoming != stored;
    case 0x206: return incoming >= stored;
    default: return true;
  }
}

struct Vtx {
  vec4 clip;
  vec4 out[VPO_COUNT];
};

struct ScreenVtx {
  float x, y, z, inv_w;
  vec4 out[VPO_COUNT];
};

// Clips a polygon in 4D homogeneous space against a plane f(clip) >= 0 (stack-allocated, max 12 verts).
constexpr int kMaxClipVerts = 12;

template <class Fn>
int clip_plane_buf(const Vtx* in, int n_in, Vtx* out, Fn eval, int clamp_w_eps = 0) {
  int n_out = 0;
  if (n_in <= 0) return 0;
  for (int i = 0; i < n_in; ++i) {
    const Vtx& a = in[i];
    const Vtx& b = in[(i + 1 == n_in) ? 0 : i + 1];
    const float fa = eval(a), fb = eval(b);
    const bool ain = fa >= 0.0f, bin = fb >= 0.0f;
    if (ain && n_out < kMaxClipVerts) out[n_out++] = a;
    if (ain != bin && n_out < kMaxClipVerts) {
      const float den = fa - fb;
      const float t = (den != 0.0f) ? std::clamp(fa / den, 0.0f, 1.0f) : 0.5f;
      Vtx& v = out[n_out++];
      for (int c = 0; c < 4; ++c) v.clip[c] = a.clip[c] + (b.clip[c] - a.clip[c]) * t;
      if (clamp_w_eps) v.clip[3] = std::max(v.clip[3], 1e-5f);
      for (int o = 0; o < VPO_COUNT; ++o)
        for (int c = 0; c < 4; ++c) v.out[o][c] = a.out[o][c] + (b.out[o][c] - a.out[o][c]) * t;
    }
  }
  return n_out;
}

// Clips a triangle in `poly[0..2]` against w >= epsilon and a screen-space guard band.
// Returns the resulting vertex count in `poly`.
int clip_frustum_buf(Vtx* poly, const float vs[3], const float vo[3],
                     uint32_t surf_w, uint32_t surf_h, uint32_t aa_x, uint32_t aa_y) {
  constexpr float kEps = 1e-5f;
  const float min_sx = -4096.0f / float(aa_x);
  const float max_sx = (float(surf_w) + 4096.0f) / float(aa_x);
  const float min_sy = -4096.0f / float(aa_y);
  const float max_sy = (float(surf_h) + 4096.0f) / float(aa_y);
  // Fast path: most triangles are completely inside the guard band.
  bool all_in = true;
  for (int i = 0; i < 3; ++i) {
    const float w = poly[i].clip[3];
    if (w < kEps || poly[i].clip[0] * vs[0] + w * (vo[0] - min_sx) < 0.0f ||
        w * (max_sx - vo[0]) - poly[i].clip[0] * vs[0] < 0.0f ||
        poly[i].clip[1] * vs[1] + w * (vo[1] - min_sy) < 0.0f ||
        w * (max_sy - vo[1]) - poly[i].clip[1] * vs[1] < 0.0f) {
      all_in = false;
      break;
    }
  }
  if (all_in) return 3;

  Vtx tmp[kMaxClipVerts];
  int n = clip_plane_buf(poly, 3, tmp, [](const Vtx& v) { return v.clip[3] - kEps; }, 1);
  if (n < 3) return n;
  n = clip_plane_buf(tmp, n, poly, [&](const Vtx& v) { return v.clip[0] * vs[0] + v.clip[3] * (vo[0] - min_sx); });
  if (n < 3) return n;
  n = clip_plane_buf(poly, n, tmp, [&](const Vtx& v) { return v.clip[3] * (max_sx - vo[0]) - v.clip[0] * vs[0]; });
  if (n < 3) return n;
  n = clip_plane_buf(tmp, n, poly, [&](const Vtx& v) { return v.clip[1] * vs[1] + v.clip[3] * (vo[1] - min_sy); });
  if (n < 3) return n;
  n = clip_plane_buf(poly, n, tmp, [&](const Vtx& v) { return v.clip[3] * (max_sy - vo[1]) - v.clip[1] * vs[1]; });
  for (int i = 0; i < n; ++i) poly[i] = tmp[i];
  return n;
}

}  // namespace

void reset_defaults(uint32_t* r) {
  r[m::COLOR_MASK] = 0x01010101;
  r[m::ALPHA_FUNC] = 0x207;  // ALWAYS
  r[m::DEPTH_FUNC] = 0x201;  // LESS
  r[m::DEPTH_MASK] = 1;
  r[m::BLEND_SFACTOR] = 0x00010001;  // ONE, ONE
  r[m::BLEND_DFACTOR] = 0;           // ZERO
  r[m::BLEND_EQUATION] = 0x80068006; // ADD
  r[m::CULL_FACE] = 0x405;           // BACK
  r[m::FRONT_FACE] = 0x901;          // CCW
  r[m::SHADER_CONTROL] = 0x40;       // 32-bit exports
  r[m::SURFACE_COLOR_TARGET] = 1;
  // Programs often clear depth without ever setting this: the reset value is the far
  // plane (depth max, stencil 0). Found running a real homebrew GCM triangle.
  r[m::ZSTENCIL_CLEAR_VALUE] = 0xFFFFFF00;
  r[m::SURFACE_FORMAT] = 8 | (2 << 5);  // A8R8G8B8, Z24S8
  for (uint32_t u = 0; u < 16; ++u) r[m::TEXTURE_BASE + 8 * u + 4] = 0x0000AAE4u;  // identity remap
}

void clear_depth(const DrawEnv& env, uint32_t mask) {
  const uint32_t* r = env.regs;
  uint32_t aa_x = 1, aa_y = 1;
  surface_aa_scale(r[m::SURFACE_FORMAT], aa_x, aa_y);
  const uint32_t clip_h = r[m::SURFACE_CLIP_H], clip_v = r[m::SURFACE_CLIP_V];
  const uint32_t w = (clip_h >> 16) * aa_x, h = (clip_v >> 16) * aa_y;
  const uint32_t x0 = (clip_h & 0xFFFF) * aa_x, y0 = (clip_v & 0xFFFF) * aa_y;
  const uint32_t depth_fmt = (r[m::SURFACE_FORMAT] >> 5) & 7, pitch = r[m::SURFACE_PITCH_Z];
  const uint32_t base = env.to_address(r[m::SURFACE_ZETA_OFFSET], r[m::CONTEXT_DMA_ZETA] == 0xFEED0001 ? 1 : 0);
  if (!base || !pitch || !w || !h) return;
  const uint32_t value = r[m::ZSTENCIL_CLEAR_VALUE];
  for (uint32_t y = y0; y < y0 + h; ++y)
    for (uint32_t x = x0; x < x0 + w; ++x) {
      uint8_t* p = env.mem + base + size_t(y) * pitch;
      if (depth_fmt == 1) {
        if (mask & 1) put_be16(p + 2 * x, uint16_t(value));
      } else {
        const uint32_t old = be32(p + 4 * x);
        uint32_t v = old;
        if (mask & 1) v = (v & 0xFF) | (value & 0xFFFFFF00u);
        if (mask & 2) v = (v & 0xFFFFFF00u) | (value & 0xFF);
        put_be32(p + 4 * x, v);
      }
    }
}

uint32_t TexUnit::cube_face_stride() const {
  const uint32_t mips = std::max(1u, mipmap);
  uint32_t total = 0;
  uint32_t mw = width, mh = height;
  const bool dxt = (format == 0x86 || format == 0x87 || format == 0x88);
  const uint32_t bpp = texel_bytes(format);
  for (uint32_t m = 0; m < mips; ++m) {
    if (dxt) total += ((mw + 3) / 4) * ((mh + 3) / 4) * (format == 0x86 ? 8u : 16u);
    else if (linear_layout) total += (pitch ? pitch : mw * bpp) * mh;
    else total += std::bit_ceil(mw) * std::bit_ceil(mh) * bpp;
    if (mw == 1 && mh == 1) break;
    mw = std::max(1u, mw >> 1);
    mh = std::max(1u, mh >> 1);
  }
  return (total + 127u) & ~127u;
}

uint32_t TexUnit::byte_size() const {
  if (cubemap) return 6u * cube_face_stride();
  const uint32_t d = is_3d() ? depth : 1u;
  if (format == 0x86 || format == 0x87 || format == 0x88)
    return ((width + 3) / 4) * ((height + 3) / 4) * d * (format == 0x86 ? 8u : 16u);
  const uint32_t bpp = texel_bytes(format);
  if (linear_layout) return (pitch ? pitch : width * bpp) * height * d;
  return std::bit_ceil(width) * std::bit_ceil(height) * std::bit_ceil(d) * bpp;
}

void decode_texture_rgba(const TexUnit& t, std::vector<uint32_t>& out) {
  if (t.cubemap) {
    const uint32_t stride = t.cube_face_stride();
    const size_t face_px = size_t(t.width) * t.height;
    out.resize(face_px * 6);
    std::vector<uint32_t> face_buf;
    TexUnit f = t;
    f.cubemap = false;
    f.dimension = 2;
    f.depth = 1;
    for (uint32_t face = 0; face < 6; ++face) {
      f.base = t.base + size_t(face) * stride;
      decode_texture_rgba(f, face_buf);
      std::memcpy(out.data() + face * face_px, face_buf.data(), face_px * 4);
    }
    return;
  }
  const uint32_t depth = t.is_3d() ? t.depth : 1u;
  out.resize(size_t(t.width) * t.height * depth);
  auto q = [](float f) { return uint32_t(std::clamp(f, 0.0f, 1.0f) * 255.0f + 0.5f); };
  if (t.is_3d()) {
    if (!t.linear_layout && (t.format == 0x85 || t.format == 0x9E)) {
      const uint32_t lw = uint32_t(std::countr_zero(std::bit_ceil(t.width)));
      const uint32_t lh = uint32_t(std::countr_zero(std::bit_ceil(t.height)));
      const uint32_t ld = uint32_t(std::countr_zero(std::bit_ceil(depth)));
      const uint32_t a_or = t.format == 0x9E ? 0xFF000000u : 0u;
      for (uint32_t z = 0; z < depth; ++z) {
        for (uint32_t y = 0; y < t.height; ++y) {
          uint32_t* dst = out.data() + (size_t(z) * t.height + y) * t.width;
          for (uint32_t x = 0; x < t.width; ++x) {
            const uint8_t* p = t.base + size_t(morton3d(x, y, z, lw, lh, ld)) * 4;
            dst[x] = uint32_t(p[1]) | (uint32_t(p[2]) << 8) | (uint32_t(p[3]) << 16) | (uint32_t(p[0]) << 24) | a_or;
          }
        }
      }
      return;
    }
    for (uint32_t z = 0; z < depth; ++z) {
      for (uint32_t y = 0; y < t.height; ++y) {
        uint32_t* dst = out.data() + (size_t(z) * t.height + y) * t.width;
        for (uint32_t x = 0; x < t.width; ++x) {
          const vec4 c = texel(t, int(x), int(y), int(z));
          dst[x] = q(c[0]) | (q(c[1]) << 8) | (q(c[2]) << 16) | (q(c[3]) << 24);
        }
      }
    }
    return;
  }
  if (t.linear_layout) {
    if (t.format == 0x81) {  // Linear B8 (e.g. Y/U/V video planes)
      const uint32_t stride = t.pitch ? t.pitch : t.width;
      for (uint32_t y = 0; y < t.height; ++y) {
        const uint8_t* row = t.base + size_t(y) * stride;
        uint32_t* dst = out.data() + size_t(y) * t.width;
        for (uint32_t x = 0; x < t.width; ++x) dst[x] = uint32_t(row[x]) * 0x01010101u;
      }
      return;
    }
    if (t.format == 0x85 || t.format == 0x9E) {  // Linear A8R8G8B8 / D8R8G8B8
      const uint32_t stride = t.pitch ? t.pitch : t.width * 4;
      const uint32_t a_or = t.format == 0x9E ? 0xFF000000u : 0u;
      for (uint32_t y = 0; y < t.height; ++y) {
        const uint8_t* row = t.base + size_t(y) * stride;
        uint32_t* dst = out.data() + size_t(y) * t.width;
        for (uint32_t x = 0; x < t.width; ++x) {
          const uint8_t* p = row + 4 * x;
          dst[x] = uint32_t(p[1]) | (uint32_t(p[2]) << 8) | (uint32_t(p[3]) << 16) | (uint32_t(p[0]) << 24) | a_or;
        }
      }
      return;
    }
  }
  if (t.format == 0x86 || t.format == 0x87 || t.format == 0x88) {
    const uint32_t bsize = t.format == 0x86 ? 8 : 16;
    const uint32_t bw = (t.width + 3) / 4, bh = (t.height + 3) / 4;
    for (uint32_t by = 0; by < bh; ++by) {
      for (uint32_t bx = 0; bx < bw; ++bx) {
        const uint8_t* blk = t.base + (size_t(by) * bw + bx) * bsize;
        const uint8_t* color = t.format == 0x86 ? blk : blk + 8;
        const uint16_t c0 = uint16_t(color[0] | (color[1] << 8)), c1 = uint16_t(color[2] | (color[3] << 8));
        const uint32_t bits = uint32_t(color[4]) | (uint32_t(color[5]) << 8) | (uint32_t(color[6]) << 16) | (uint32_t(color[7]) << 24);
        const vec4 a = unpack565(c0), b = unpack565(c1);
        const bool four = t.format != 0x86 || c0 > c1;
        uint32_t pal[4];
        pal[0] = q(a[0]) | (q(a[1]) << 8) | (q(a[2]) << 16) | 0xFF000000u;
        pal[1] = q(b[0]) | (q(b[1]) << 8) | (q(b[2]) << 16) | 0xFF000000u;
        if (four) {
          pal[2] = q(a[0] + (b[0] - a[0]) * (1.0f / 3)) | (q(a[1] + (b[1] - a[1]) * (1.0f / 3)) << 8) |
                   (q(a[2] + (b[2] - a[2]) * (1.0f / 3)) << 16) | 0xFF000000u;
          pal[3] = q(a[0] + (b[0] - a[0]) * (2.0f / 3)) | (q(a[1] + (b[1] - a[1]) * (2.0f / 3)) << 8) |
                   (q(a[2] + (b[2] - a[2]) * (2.0f / 3)) << 16) | 0xFF000000u;
        } else {
          pal[2] = q((a[0] + b[0]) * 0.5f) | (q((a[1] + b[1]) * 0.5f) << 8) | (q((a[2] + b[2]) * 0.5f) << 16) | 0xFF000000u;
          pal[3] = 0;
        }
        uint32_t apal[8];
        uint64_t abits = 0;
        if (t.format == 0x88) {
          const float a0 = blk[0] / 255.0f, a1 = blk[1] / 255.0f;
          for (int k = 0; k < 6; ++k) abits |= uint64_t(blk[2 + k]) << (8 * k);
          apal[0] = blk[0];
          apal[1] = blk[1];
          if (blk[0] > blk[1]) {
            for (int s = 2; s < 8; ++s) apal[s] = q((a0 * (8 - s) + a1 * (s - 1)) / 7.0f);
          } else {
            for (int s = 2; s < 6; ++s) apal[s] = q((a0 * (6 - s) + a1 * (s - 1)) / 5.0f);
            apal[6] = 0;
            apal[7] = 255;
          }
        }
        for (uint32_t py = 0; py < 4; ++py) {
          const uint32_t y = by * 4 + py;
          if (y >= t.height) break;
          uint32_t* dst = out.data() + size_t(y) * t.width;
          for (uint32_t px = 0; px < 4; ++px) {
            const uint32_t x = bx * 4 + px;
            if (x >= t.width) break;
            const uint32_t i = py * 4 + px;
            uint32_t rgba = pal[(bits >> (2 * i)) & 3];
            if (t.format == 0x87) {
              const uint32_t a4 = (blk[i / 2] >> ((i & 1) * 4)) & 0xF;
              rgba = (rgba & 0x00FFFFFFu) | ((a4 * 17u) << 24);
            } else if (t.format == 0x88) {
              rgba = (rgba & 0x00FFFFFFu) | (apal[(abits >> (3 * i)) & 7] << 24);
            }
            dst[x] = rgba;
          }
        }
      }
    }
    return;
  }
  for (uint32_t y = 0; y < t.height; ++y)
    for (uint32_t x = 0; x < t.width; ++x) {
      const vec4 c = texel(t, int(x), int(y));
      out[size_t(y) * t.width + x] = q(c[0]) | (q(c[1]) << 8) | (q(c[2]) << 16) | (q(c[3]) << 24);
    }
}

bool prepare_draw_state(const DrawEnv& env, PreparedDraw& d, std::string* error) {
  const uint32_t* r = env.regs;
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
  d.tris.clear();
  d.fp_consts.clear();
  d.raw_tris = d.rej_nonfinite = d.rej_clip = d.rej_cull = 0;

  // ---- render target ----
  const uint32_t surf = r[m::SURFACE_FORMAT];
  const uint32_t color_fmt = surf & 0x1F, surf_type = (surf >> 8) & 0xF;
  d.depth_fmt = (surf >> 5) & 7;
  if (surf_type == 2) return fail("swizzled render targets are not supported yet");
  const bool color_off = (r[m::COLOR_MASK] == 0 || r[m::SURFACE_COLOR_TARGET] == 0);
  d.argb = color_fmt == 8 || color_fmt == 4 || color_fmt == 5;
  const bool abgr = color_fmt == 16 || color_fmt == 14 || color_fmt == 15;
  if (!color_off && !d.argb && !abgr) return fail("only 32-bit colour render targets are supported yet");
  d.pitch = r[m::SURFACE_PITCH_A];
  d.color_base = env.to_address(r[m::SURFACE_COLOR_AOFFSET], r[m::CONTEXT_DMA_COLOR_A] == 0xFEED0001 ? 1 : 0);
  if (!color_off && (!d.color_base || !d.pitch)) return fail("no colour surface bound");
  d.zpitch = r[m::SURFACE_PITCH_Z];
  d.zbase = env.to_address(r[m::SURFACE_ZETA_OFFSET], r[m::CONTEXT_DMA_ZETA] == 0xFEED0001 ? 1 : 0);
  if (color_off && (!d.zbase || !d.zpitch)) return fail("no depth surface bound");

  const uint32_t clip_h = r[m::SURFACE_CLIP_H], clip_v = r[m::SURFACE_CLIP_V];
  uint32_t aa_x = 1, aa_y = 1;
  surface_aa_scale(surf, aa_x, aa_y);
  d.aa_x = aa_x;
  d.aa_y = aa_y;
  d.min_x = int(clip_h & 0xFFFF) * int(aa_x);
  d.max_x = d.min_x + int(clip_h >> 16) * int(aa_x);
  d.min_y = int(clip_v & 0xFFFF) * int(aa_y);
  d.max_y = d.min_y + int(clip_v >> 16) * int(aa_y);
  d.surface_w = uint32_t(d.max_x);
  d.surface_h = uint32_t(d.max_y);
  if (r[m::SCISSOR_H] >> 16) {
    d.min_x = std::max(d.min_x, int(r[m::SCISSOR_H] & 0xFFFF) * int(aa_x));
    d.max_x = std::min(d.max_x, (int(r[m::SCISSOR_H] & 0xFFFF) + int(r[m::SCISSOR_H] >> 16)) * int(aa_x));
  }
  if (r[m::SCISSOR_V] >> 16) {
    d.min_y = std::max(d.min_y, int(r[m::SCISSOR_V] & 0xFFFF) * int(aa_y));
    d.max_y = std::min(d.max_y, (int(r[m::SCISSOR_V] & 0xFFFF) + int(r[m::SCISSOR_V] >> 16)) * int(aa_y));
  }
  const uint32_t zbpp = d.depth_fmt == 1 ? 2 : 4;
  const uint32_t max_w = (color_off && d.zbase && d.zpitch >= zbpp) ? (d.zpitch / zbpp) : (d.pitch / 4);
  d.max_x = std::min(d.max_x, int(max_w));
  d.surface_w = std::min(d.surface_w, max_w);
  if (d.min_x >= d.max_x || d.min_y >= d.max_y) return true;

  for (int k = 0; k < 3; ++k) {
    d.vp_scale[k] = fbits(r[m::VIEWPORT_SCALE + k]);
    d.vp_offset[k] = fbits(r[m::VIEWPORT_OFFSET + k]);
  }

  // ---- fragment program ----
  const uint32_t sp = r[m::SHADER_PROGRAM];
  const uint32_t fp_addr = env.to_address(sp & ~3u, (sp & 3) == 2 ? 1 : 0);
  if (!fp_addr) return fail("no fragment program bound");
  d.fp = decode_fp(env.mem + fp_addr);
  d.fp32_out = (r[m::SHADER_CONTROL] & 0x40) != 0;
  {
    // Hash ONLY instruction words (excluding inline float constants, which are
    // supplied via UBO binding 16), so animating shader constants never triggers
    // SPIR-V / Vulkan pipeline recompilation.
    uint64_t h = 1469598103934665603ull;
    size_t off = 0;
    for (const FpInsn& i : d.fp) {
      uint64_t w01 = 0, w23 = 0;
      std::memcpy(&w01, env.mem + fp_addr + off, 8);
      std::memcpy(&w23, env.mem + fp_addr + off + 8, 8);
      h = (h ^ w01) * 1099511628211ull;
      h = (h ^ w23) * 1099511628211ull;
      off += i.has_constant ? 32 : 16;
      if (i.has_constant) d.fp_consts.push_back(i.constant);
      if (i.end) break;
    }
    d.fp_hash = h ^ (d.fp32_out ? 0x9E37u : 0);
  }

  // ---- textures ----
  uint32_t used_tex = 0;
  for (const FpInsn& i : d.fp)
    if (i.op == FP_TEX || i.op == FP_TXP || i.op == FP_TXL || i.op == FP_TXB || i.op == FP_TXD)
      used_tex |= 1u << (i.tex & 15);
  for (uint32_t u = 0; u < 16; ++u) {
    const uint32_t* t = r + m::TEXTURE_BASE + 8 * u;
    TexUnit& tu = d.tex[u];
    tu = TexUnit{};
    tu.enabled = ((t[3] >> 31) & 1) && ((used_tex & (1u << u)) != 0);
    if (!tu.enabled) continue;
    const uint32_t loc = (t[1] & 3) == 2 ? 1 : 0;
    const uint32_t gcm = (t[1] >> 8) & 0xFF;
    tu.format = gcm & ~0x60u;
    tu.linear_layout = (gcm & 0x20) != 0;
    tu.unnormalized = (gcm & 0x40) != 0;
    tu.cubemap = ((t[1] >> 2) & 1) != 0;
    tu.dimension = (t[1] >> 4) & 0xF;
    if (tu.dimension == 0) tu.dimension = 2;
    tu.mipmap = std::max(1u, (t[1] >> 16) & 0xF);
    tu.width = std::max(1u, t[6] >> 16);
    tu.height = std::max(1u, t[6] & 0xFFFF);
    tu.depth = tu.is_3d() ? std::max(1u, (r[m::TEXTURE_CONTROL3 + u] >> 20) & 0xFFFu) : 1u;
    if (tu.linear_layout) tu.pitch = r[m::TEXTURE_CONTROL3 + u] & 0xFFFFF;  // row pitch in bytes
    static constexpr uint32_t kSupported[] = {0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x8B, 0x8F, 0x90, 0x91,
                                              0x92, 0x94, 0x95, 0x97, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F};
    if (std::find(std::begin(kSupported), std::end(kSupported), tu.format) == std::end(kSupported)) {
      static std::atomic<uint64_t> reported[4];  // one bit per format value
      const uint64_t bit = 1ull << (tu.format & 63);
      if (!(reported[(tu.format >> 6) & 3].fetch_or(bit) & bit))
        std::fprintf(stderr, "[rsx] texture format 0x%02X (%ux%u, %s) is not supported yet: sampled as black\n",
                     tu.format, tu.width, tu.height, tu.linear_layout ? "linear" : "swizzled");
    }
    tu.wrap_s = t[2] & 0xF;
    tu.wrap_t = (t[2] >> 8) & 0xF;
    tu.wrap_r = (t[2] >> 16) & 0xF;
    tu.gamma = (t[2] >> 20) & 0xF;
    tu.zfunc = (t[2] >> 28) & 7;
    tu.remap = t[4] & 0xFFFFu;
    tu.border_color = t[7];
    const uint32_t mag_f = (t[5] >> 24) & 7;
    const uint32_t min_f = (t[5] >> 16) & 7;
    tu.signed_mask = uint8_t(((t[5] >> 29) & 1u) | (((t[5] >> 30) & 1u) << 1) |
                             (((t[5] >> 31) & 1u) << 2) | (((t[5] >> 28) & 1u) << 3));
    tu.mag_linear = (mag_f == 2 || mag_f == 4);
    tu.min_linear = (min_f == 2 || min_f == 4 || min_f == 6 || min_f == 7);
    tu.use_mip = (min_f >= 3 && min_f <= 6) && tu.mipmap > 1;
    tu.mip_linear = (min_f == 5 || min_f == 6);
    tu.lod_bias = float(int32_t((t[5] & 0x1FFFu) << 19) >> 19) / 256.0f;
    tu.bilinear = tu.mag_linear || tu.min_linear;
    tu.address = env.to_address(t[0], loc);
    tu.base = tu.address ? env.mem + tu.address : nullptr;
  }

  // ---- fixed-function state ----
  d.depth_test = r[m::DEPTH_TEST_ENABLE] && d.zbase && d.zpitch;
  d.depth_write = d.depth_test && r[m::DEPTH_MASK];
  d.depth_func = r[m::DEPTH_FUNC];
  d.blend = r[m::BLEND_ENABLE] != 0;
  d.sf_rgb = r[m::BLEND_SFACTOR] & 0xFFFF;
  d.sf_a = r[m::BLEND_SFACTOR] >> 16;
  d.df_rgb = r[m::BLEND_DFACTOR] & 0xFFFF;
  d.df_a = r[m::BLEND_DFACTOR] >> 16;
  d.eq_rgb = r[m::BLEND_EQUATION] & 0xFFFF;
  d.eq_a = r[m::BLEND_EQUATION] >> 16;
  d.color_mask = (r[m::SURFACE_COLOR_TARGET] == 0) ? 0u : r[m::COLOR_MASK];
  d.alpha_test = r[m::ALPHA_TEST_ENABLE] != 0;
  d.alpha_func = r[m::ALPHA_FUNC];
  d.alpha_ref = uint8_t(r[m::ALPHA_REF] & 0xFF);
  d.blend_color = r[m::BLEND_COLOR];
  d.cull = r[m::CULL_FACE_ENABLE] != 0;
  d.cull_face = r[m::CULL_FACE];
  d.front_ccw = r[m::FRONT_FACE] != 0x900;
  d.poly_offset = r[m::POLY_OFFSET_FILL_ENABLE] != 0;
  d.poly_offset_scale = fbits(r[m::POLYGON_OFFSET_SCALE_FACTOR]);
  d.poly_offset_bias = fbits(r[m::POLYGON_OFFSET_BIAS]);
  return true;
}

bool pack_indexed_geometry(const DrawEnv& env, const DrawCall& call, uint32_t in_mask, uint32_t in_count,
                           float* dst_vtx, uint32_t max_vtx, uint32_t& out_vtx,
                           uint32_t* dst_idx, uint32_t max_idx, uint32_t& out_idx, std::string* error) {
  const uint32_t* r = env.regs;
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
  out_vtx = out_idx = 0;

  if (call.primitive < 5 || call.primitive > 10) return fail("points and lines are not supported yet");

  size_t n_raw = 0;
  for (const auto& [first, count] : call.ranges) n_raw += count;
  if (n_raw < 3) return true;

  size_t need_idx = 0;
  switch (call.primitive) {
    case 5: need_idx = (n_raw / 3) * 3; break;
    case 6: case 7: case 10: need_idx = (n_raw - 2) * 3; break;
    case 8: need_idx = (n_raw / 4) * 6; break;
    case 9: need_idx = n_raw >= 4 ? ((n_raw - 2) / 2) * 6 : 0; break;
    default: break;
  }
  if (need_idx == 0) return true;
  if (need_idx > max_idx) return fail("index ring overflow");

  // Resolve only the attributes actually read by the vertex program.
  const uint32_t base_off = r[m::VERTEX_DATA_BASE_OFFSET];
  const uint32_t base_idx = r[m::VERTEX_DATA_BASE_INDEX];
  Attrib attribs[16];
  uint8_t active[16];
  uint32_t n_act = 0;
  for (uint32_t i = 0; i < 16; ++i) {
    if (!(in_mask & (1u << i))) continue;
    active[n_act++] = uint8_t(i);
    const uint32_t f = r[m::VERTEX_ARRAY_FORMAT + i], off = r[m::VERTEX_ARRAY_OFFSET + i];
    attribs[i].type = f & 0xF;
    attribs[i].size = (f >> 4) & 0xF;
    attribs[i].stride = (f >> 8) & 0xFF;
    if (attribs[i].size) {
      const uint32_t a = env.to_address((off & 0x7FFFFFFF) + base_off, off >> 31);
      attribs[i].base = a ? env.mem + a : nullptr;
    }
  }
  const uint32_t vtx_floats = in_count * 4;

  // Compact L1-friendly remap table: guest vertex id -> packed local vertex index.
  struct IdSlot { uint32_t stamp = 0, key = 0, local = 0; };
  thread_local std::vector<IdSlot> rtab;
  thread_local uint32_t rstamp = 0;
  thread_local std::vector<uint32_t> local_ids;
  const size_t want_cap = std::bit_ceil(std::max<size_t>(256, n_raw * 2 + 16));
  if (rtab.size() < want_cap) {
    rtab.assign(want_cap, IdSlot{});
    rstamp = 0;
  }
  if (++rstamp == 0) {
    for (auto& sl : rtab) sl.stamp = 0;
    rstamp = 1;
  }
  const size_t rmask = rtab.size() - 1;
  local_ids.clear();
  local_ids.reserve(n_raw);

  auto intern = [&](uint32_t id) -> uint32_t {
    size_t slot = (size_t(id) * 2654435761u) & rmask;
    while (rtab[slot].stamp == rstamp) {
      if (rtab[slot].key == id) return rtab[slot].local;
      slot = (slot + 1) & rmask;
    }
    const uint32_t loc = out_vtx++;
    if (loc < max_vtx && vtx_floats) {
      float* dst = dst_vtx + size_t(loc) * vtx_floats;
      for (uint32_t ai = 0; ai < n_act; ++ai) {
        const uint32_t attr_i = active[ai];
        const vec4 v = attribs[attr_i].size ? fetch_attrib(attribs[attr_i], id)
                                            : (env.const_attribs ? env.const_attribs[attr_i] : vec4{0, 0, 0, 1});
        dst[0] = v[0]; dst[1] = v[1]; dst[2] = v[2]; dst[3] = v[3];
        dst += 4;
      }
    }
    rtab[slot] = {rstamp, id, loc};
    return loc;
  };

  if (call.indexed) {
    const uint32_t dma = r[m::INDEX_ARRAY_DMA];
    const uint32_t ib = env.to_address(r[m::INDEX_ARRAY_ADDRESS], dma & 0xF);
    if (!ib) return fail("no index buffer");
    const bool u16 = ((dma >> 4) & 0xF) == 1;
    const uint8_t* ibp = env.mem + ib;
    for (const auto& [first, count] : call.ranges) {
      for (uint32_t k = 0; k < count; ++k) {
        const uint8_t* p = ibp + size_t(first + k) * (u16 ? 2 : 4);
        local_ids.push_back(intern((u16 ? be16(p) : be32(p)) + base_idx));
      }
    }
  } else {
    for (const auto& [first, count] : call.ranges)
      for (uint32_t k = 0; k < count; ++k) local_ids.push_back(intern(first + k));
  }
  if (out_vtx > max_vtx) return fail("vertex ring overflow");

  // Expand primitive into triangle-list indices.
  const size_t n = local_ids.size();
  uint32_t* o = dst_idx;
  switch (call.primitive) {
    case 5:
      for (size_t i = 0; i + 2 < n; i += 3) { *o++ = local_ids[i]; *o++ = local_ids[i + 1]; *o++ = local_ids[i + 2]; }
      break;
    case 6:
      for (size_t i = 0; i + 2 < n; ++i) {
        if (i & 1) { *o++ = local_ids[i + 1]; *o++ = local_ids[i]; *o++ = local_ids[i + 2]; }
        else       { *o++ = local_ids[i]; *o++ = local_ids[i + 1]; *o++ = local_ids[i + 2]; }
      }
      break;
    case 7: case 10:
      for (size_t i = 1; i + 1 < n; ++i) { *o++ = local_ids[0]; *o++ = local_ids[i]; *o++ = local_ids[i + 1]; }
      break;
    case 8:
      for (size_t i = 0; i + 3 < n; i += 4) {
        *o++ = local_ids[i]; *o++ = local_ids[i + 1]; *o++ = local_ids[i + 2];
        *o++ = local_ids[i]; *o++ = local_ids[i + 2]; *o++ = local_ids[i + 3];
      }
      break;
    case 9:
      for (size_t i = 0; i + 3 < n; i += 2) {
        *o++ = local_ids[i]; *o++ = local_ids[i + 1]; *o++ = local_ids[i + 3];
        *o++ = local_ids[i]; *o++ = local_ids[i + 3]; *o++ = local_ids[i + 2];
      }
      break;
    default: break;
  }
  out_idx = uint32_t(o - dst_idx);
  return true;
}

bool prepare_draw(const DrawEnv& env, const DrawCall& call, PreparedDraw& d, std::string* error) {
  if (!prepare_draw_state(env, d, error)) return false;
  if (d.min_x >= d.max_x || d.min_y >= d.max_y) return true;
  const uint32_t* r = env.regs;
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };

  // ---- vertex arrays ----
  const uint32_t base_off = r[m::VERTEX_DATA_BASE_OFFSET];
  const uint32_t base_idx = r[m::VERTEX_DATA_BASE_INDEX];
  Attrib attribs[16];
  for (int i = 0; i < 16; ++i) {
    const uint32_t f = r[m::VERTEX_ARRAY_FORMAT + i], off = r[m::VERTEX_ARRAY_OFFSET + i];
    attribs[i].type = f & 0xF;
    attribs[i].size = (f >> 4) & 0xF;
    attribs[i].stride = (f >> 8) & 0xFF;
    if (attribs[i].size) {
      const uint32_t a = env.to_address((off & 0x7FFFFFFF) + base_off, off >> 31);
      attribs[i].base = a ? env.mem + a : nullptr;
    }
  }

  // ---- vertex ids in submission order ----
  thread_local std::vector<uint32_t> ids;
  ids.clear();
  if (call.indexed) {
    const uint32_t dma = r[m::INDEX_ARRAY_DMA];
    const uint32_t ib = env.to_address(r[m::INDEX_ARRAY_ADDRESS], dma & 0xF);
    if (!ib) return fail("no index buffer");
    const bool u16 = ((dma >> 4) & 0xF) == 1;
    for (const auto& [first, count] : call.ranges)
      for (uint32_t k = 0; k < count; ++k) {
        const uint8_t* p = env.mem + ib + size_t(first + k) * (u16 ? 2 : 4);
        ids.push_back((u16 ? be16(p) : be32(p)) + base_idx);
      }
  } else {
    for (const auto& [first, count] : call.ranges)
      for (uint32_t k = 0; k < count; ++k) ids.push_back(first + k);
  }

  // ---- vertex shading (cached per vertex id in a reusable flat table) ----
  struct VtxSlot { uint32_t stamp = 0, key = 0; Vtx val; };
  thread_local std::vector<VtxSlot> vcache;
  thread_local uint32_t vstamp = 0;
  const size_t want_cap = std::bit_ceil(std::max<size_t>(256, ids.size() * 2 + 16));
  if (vcache.size() < want_cap) {
    vcache.assign(want_cap, VtxSlot{});
    vstamp = 0;
  }
  if (++vstamp == 0) {
    for (auto& sl : vcache) sl.stamp = 0;
    vstamp = 1;
  }
  const size_t vmask = vcache.size() - 1;
  std::string vp_error;
  auto vertex = [&](uint32_t id) -> const Vtx* {
    size_t idx = (size_t(id) * 2654435761u) & vmask;
    while (vcache[idx].stamp == vstamp) {
      if (vcache[idx].key == id) return &vcache[idx].val;
      idx = (idx + 1) & vmask;
    }
    vec4 in[16];
    for (int i = 0; i < 16; ++i)
      in[i] = attribs[i].size ? fetch_attrib(attribs[i], id)
                              : (env.const_attribs ? env.const_attribs[i] : vec4{0, 0, 0, 1});
    VtxSlot& sl = vcache[idx];
    if (!run_vp(*env.vp, in, sl.val.out, &vp_error)) return nullptr;
    sl.val.clip = sl.val.out[VPO_HPOS];
    sl.key = id;
    sl.stamp = vstamp;
    return &sl.val;
  };

  // ---- primitive assembly into triangles (index triples into `ids`) ----
  thread_local std::vector<std::array<size_t, 3>> tris;
  tris.clear();
  const size_t n = ids.size();
  switch (call.primitive) {
    case 5: for (size_t i = 0; i + 2 < n; i += 3) tris.push_back({i, i + 1, i + 2}); break;
    case 6:
      for (size_t i = 0; i + 2 < n; ++i)
        tris.push_back(i % 2 ? std::array<size_t, 3>{i + 1, i, i + 2} : std::array<size_t, 3>{i, i + 1, i + 2});
      break;
    case 7: case 10:
      for (size_t i = 1; i + 1 < n; ++i) tris.push_back({0, i, i + 1});
      break;
    case 8:
      for (size_t i = 0; i + 3 < n; i += 4) { tris.push_back({i, i + 1, i + 2}); tris.push_back({i, i + 2, i + 3}); }
      break;
    case 9:
      for (size_t i = 0; i + 3 < n; i += 2) { tris.push_back({i, i + 1, i + 3}); tris.push_back({i, i + 3, i + 2}); }
      break;
    default: return fail("points and lines are not supported yet");
  }

  // ---- clip, project, cull (stack-allocated, zero heap allocations per tri) ----
  const float* vs = d.vp_scale;
  const float* vo = d.vp_offset;
  const uint32_t aa_x = d.aa_x, aa_y = d.aa_y;
  auto fast_finite = [](float x) { return (std::bit_cast<uint32_t>(x) & 0x7F800000u) != 0x7F800000u; };
  Vtx poly[kMaxClipVerts];
  PreparedVertex sv[kMaxClipVerts];
  for (const auto& tri : tris) {
    ++d.raw_tris;
    bool finite_tri = true;
    for (int vi = 0; vi < 3; ++vi) {
      const Vtx* v = vertex(ids[tri[vi]]);
      if (!v) return fail(vp_error);
      if (d.raw_tris == 1)
        for (int c = 0; c < 4; ++c) d.first_clip[vi][c] = v->clip[c];
      for (int c = 0; c < 4; ++c)
        if (!fast_finite(v->clip[c])) finite_tri = false;
      poly[vi] = *v;
    }
    if (!finite_tri) { ++d.rej_nonfinite; continue; }
    const int n_poly = clip_frustum_buf(poly, vs, vo, d.surface_w, d.surface_h, aa_x, aa_y);
    if (n_poly < 3) { ++d.rej_clip; continue; }

    for (int i = 0; i < n_poly; ++i) {
      const float iw = 1.0f / poly[i].clip[3];
      sv[i].x = (poly[i].clip[0] * iw * vs[0] + vo[0]) * float(aa_x);
      sv[i].y = (poly[i].clip[1] * iw * vs[1] + vo[1]) * float(aa_y);
      sv[i].z = poly[i].clip[2] * iw * vs[2] + vo[2];
      sv[i].inv_w = iw;
      sv[i].vary[0] = poly[i].out[VPO_COL0];
      sv[i].vary[1] = poly[i].out[VPO_COL1];
      sv[i].vary[2] = poly[i].out[VPO_FOG];
      for (int k = 0; k < 10; ++k) sv[i].vary[3 + k] = poly[i].out[VPO_TEX0 + k];
    }
    bool kept = false;
    for (int t = 1; t + 1 < n_poly; ++t) {
      const PreparedVertex& a = sv[0];
      const PreparedVertex& b = sv[t];
      const PreparedVertex& c = sv[t + 1];
      const float area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
      if (area == 0.0f) continue;
      // Window y grows downwards, so a negative area is counter-clockwise on screen.
      const bool ccw = area < 0;
      const bool front = d.front_ccw ? ccw : !ccw;
      if (d.cull && ((d.cull_face == 0x405 && !front) || (d.cull_face == 0x404 && front) || d.cull_face == 0x408)) continue;
      d.tris.push_back(a);
      d.tris.push_back(b);
      d.tris.push_back(c);
      kept = true;
    }
    if (!kept) ++d.rej_cull;
  }
  return true;
}

bool raster_cpu(const DrawEnv& env, const PreparedDraw& d, DrawStats& stats, std::string* error) {
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
  PixelEnv px;
  for (int u = 0; u < 16; ++u) px.tex[u] = d.tex[u];

  uint32_t write_mask = 0;
  if (d.color_mask & 0x01000000) write_mask |= 0xFF000000u;
  if (d.color_mask & 0x00010000) write_mask |= d.argb ? 0x00FF0000u : 0x000000FFu;
  if (d.color_mask & 0x00000100) write_mask |= 0x0000FF00u;
  if (d.color_mask & 0x00000001) write_mask |= d.argb ? 0x000000FFu : 0x00FF0000u;

  std::string fp_error;
  for (size_t t = 0; t + 2 < d.tris.size(); t += 3) {
    const PreparedVertex& a = d.tris[t];
    const PreparedVertex& b = d.tris[t + 1];
    const PreparedVertex& c = d.tris[t + 2];
    ++stats.triangles;

    const int bx0 = std::max(d.min_x, int(std::floor(std::min({a.x, b.x, c.x}))));
    const int bx1 = std::min(d.max_x - 1, int(std::ceil(std::max({a.x, b.x, c.x}))));
    const int by0 = std::max(d.min_y, int(std::floor(std::min({a.y, b.y, c.y}))));
    const int by1 = std::min(d.max_y - 1, int(std::ceil(std::max({a.y, b.y, c.y}))));

    // Edge functions with a consistent orientation (positive inside) and a fill
    // rule: a pixel centre exactly on an edge belongs to only one of the two
    // triangles sharing that edge, so shared edges are neither drawn twice
    // (which would double-blend) nor skipped.
    const PreparedVertex* A = &a;
    const PreparedVertex* B = &b;
    const PreparedVertex* C = &c;
    auto edge = [](const PreparedVertex* p, const PreparedVertex* q, float x, float y) {
      return (q->x - p->x) * (y - p->y) - (q->y - p->y) * (x - p->x);
    };
    float oriented = edge(A, B, C->x, C->y);
    if (oriented == 0) continue;
    if (oriented < 0) {
      std::swap(B, C);
      oriented = -oriented;
    }
    // Top-left rule (as GPUs implement it). Triangles are normalised to clockwise on
    // screen (y down): left edges run upwards (dy < 0), top edges run rightwards.
    auto owns = [](const PreparedVertex* p, const PreparedVertex* q) {
      const float dx = q->x - p->x, dy = q->y - p->y;
      return dy < 0 || (dy == 0 && dx > 0);
    };
    const bool own_a = owns(B, C), own_b = owns(C, A), own_c = owns(A, B);
    const float inv_area = 1.0f / oriented;

    for (int y = by0; y <= by1; ++y) {
      for (int x = bx0; x <= bx1; ++x) {
        const float pxc = x + 0.5f, pyc = y + 0.5f;
        const float ea = edge(B, C, pxc, pyc), eb = edge(C, A, pxc, pyc), ec = edge(A, B, pxc, pyc);
        if (ea < 0 || eb < 0 || ec < 0) continue;
        if ((ea == 0 && !own_a) || (eb == 0 && !own_b) || (ec == 0 && !own_c)) continue;
        // Barycentric weights for the original vertex order a, b, c.
        const float wa = ea * inv_area, wb = eb * inv_area, wc = ec * inv_area;
        const float w0 = wa;
        const float w1 = B == &b ? wb : wc;
        const float w2 = B == &b ? wc : wb;

        const float z = w0 * a.z + w1 * b.z + w2 * c.z;
        uint8_t* zp = d.zbase ? env.mem + d.zbase + size_t(y) * d.zpitch : nullptr;
        uint32_t zval = 0;
        if (d.depth_test) {
          if (d.depth_fmt == 1) {
            zval = uint32_t(std::clamp(z, 0.0f, 1.0f) * 65535.0f + 0.5f);
            if (!depth_pass(d.depth_func, zval, be16(zp + 2 * x))) continue;
          } else {
            zval = uint32_t(std::clamp(z, 0.0f, 1.0f) * 16777215.0f + 0.5f);
            if (!depth_pass(d.depth_func, zval, be32(zp + 4 * x) >> 8)) continue;
          }
        }

        // Perspective-correct varyings.
        const float iw = w0 * a.inv_w + w1 * b.inv_w + w2 * c.inv_w;
        auto vary = [&](int o) {
          vec4 v;
          for (int k = 0; k < 4; ++k)
            v[k] = (w0 * a.vary[o][k] * a.inv_w + w1 * b.vary[o][k] * b.inv_w + w2 * c.vary[o][k] * c.inv_w) / iw;
          return v;
        };
        vec4 in[FPI_COUNT]{};
        in[FPI_WPOS] = {pxc, pyc, z, iw};
        in[FPI_COL0] = vary(0);
        in[FPI_COL1] = vary(1);
        in[FPI_FOG] = vary(2);
        for (int k = 0; k < 10; ++k) in[FPI_TEX0 + k] = vary(3 + k);

        FpResult res;
        if (!run_fp(d.fp, in, d.fp32_out, sample, &px, res, &fp_error)) return fail(fp_error);
        if (res.discard) continue;

        vec4 s = res.color;
        for (float& ch : s) ch = std::clamp(ch, 0.0f, 1.0f);
        if (d.alpha_test && d.alpha_func != 0x207) {
          const float ref = float(d.alpha_ref) / 255.0f;
          bool pass = true;
          switch (d.alpha_func) {
            case 0x200: pass = false; break;
            case 0x201: pass = s[3] < ref; break;
            case 0x202: pass = s[3] == ref; break;
            case 0x203: pass = s[3] <= ref; break;
            case 0x204: pass = s[3] > ref; break;
            case 0x205: pass = s[3] != ref; break;
            case 0x206: pass = s[3] >= ref; break;
            default: break;
          }
          if (!pass) continue;
        }
        if (write_mask) {
          uint8_t* cp = env.mem + d.color_base + size_t(y) * d.pitch + size_t(x) * 4;
          const uint32_t old = be32(cp);
          vec4 dst;
          if (d.argb) dst = {((old >> 16) & 0xFF) / 255.0f, ((old >> 8) & 0xFF) / 255.0f, (old & 0xFF) / 255.0f, (old >> 24) / 255.0f};
          else dst = {(old & 0xFF) / 255.0f, ((old >> 8) & 0xFF) / 255.0f, ((old >> 16) & 0xFF) / 255.0f, (old >> 24) / 255.0f};
          if (d.blend) {
            vec4 o;
            for (int k = 0; k < 3; ++k)
              o[k] = equation(d.eq_rgb, s[k] * factor(d.sf_rgb, s, dst, k), dst[k] * factor(d.df_rgb, s, dst, k));
            o[3] = equation(d.eq_a, s[3] * factor(d.sf_a, s, dst, 3), dst[3] * factor(d.df_a, s, dst, 3));
            s = o;
          }
          auto q = [](float f) { return uint32_t(std::clamp(f, 0.0f, 1.0f) * 255.0f + 0.5f); };
          const uint32_t packed = d.argb ? (q(s[3]) << 24) | (q(s[0]) << 16) | (q(s[1]) << 8) | q(s[2])
                                         : (q(s[3]) << 24) | (q(s[2]) << 16) | (q(s[1]) << 8) | q(s[0]);
          put_be32(cp, (old & ~write_mask) | (packed & write_mask));
        }
        if (d.depth_write) {
          if (d.depth_fmt == 1) put_be16(zp + 2 * x, uint16_t(zval));
          else put_be32(zp + 4 * x, (zval << 8) | (be32(zp + 4 * x) & 0xFF));
        }
        ++stats.pixels;
      }
    }
  }
  return true;
}

bool draw(const DrawEnv& env, const DrawCall& call, DrawStats& stats, std::string* error) {
  PreparedDraw d;
  if (!prepare_draw(env, call, d, error)) return false;
  return raster_cpu(env, d, stats, error);
}

}  // namespace rsx
