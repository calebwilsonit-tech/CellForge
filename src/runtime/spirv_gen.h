#pragma once
// SPIR-V generation for the GPU backend: a passthrough vertex shader and a
// translator from decoded RSX fragment programs to Vulkan fragment shaders.
//
// Interface shared by both shaders:
//   vertex inputs  location 0      = clip-space position
//                  location 1..13  = varyings (COL0, COL1, FOG, TEX0..TEX9)
//   vertex outputs / fragment inputs location 0..12 = the same varyings
//   fragment output location 0 = colour
//   textures: descriptor set 0, binding = RSX texture unit (combined image samplers)
//   fragment inline constants: descriptor set 0, binding 16 (std140 vec4 array UBO)

#include <cstdint>
#include <string>
#include <vector>

#include "rsx_program.h"

namespace rsx {

std::vector<uint32_t> spirv_passthrough_vs();
// GPU format-reinterpretation fragment shaders (used with spirv_passthrough_vs,
// sampling binding 0 at Location 3 TEX0.xy):
// - D32_SFLOAT depth -> packed D24S8 A8R8G8B8 (matches write_back + texel 0x85)
std::vector<uint32_t> spirv_convert_d24_to_argb_fs();
// - A8B8G8R8 color (R8G8B8A8_UNORM) -> Y16_X16 (R16G16_UNORM, matches write_back + texel 0x95)
std::vector<uint32_t> spirv_convert_abgr_to_y16x16_fs();

// Metadata describing the vertex inputs and constants required by a translated VP.
struct SpirvVpMeta {
  uint32_t in_mask = 0;         // bit k set if RSX vertex attribute k (0..15) is read
  uint8_t in_loc[16]{};         // packed Vulkan input location for each used attribute
  uint32_t in_count = 0;        // number of packed vec4 input attributes
  bool uses_rel_const = false;  // true if any instruction uses ARL relative addressing
  uint32_t max_const = 0;       // highest VP constant index referenced (511 if uses_rel_const)
};

// Translates a decoded RSX vertex program (starting at vp.start) to a Vulkan SPIR-V
// vertex shader. Reads packed vec4 inputs at Location 0..(in_count-1) and a std140 UBO
// at set 0, binding 17: { vec4 vp_scale; vec4 vp_offset; vec4 c[512]; }.
// Returns false for programs using flow control or unsupported ops (caller falls back to CPU VP).
bool spirv_translate_vp(const VpState& vp, std::vector<uint32_t>& out, SpirvVpMeta& meta, std::string* error);

// Returns false (with a reason) for programs using instructions the translator
// does not handle; the caller then falls back to the CPU renderer for that draw.
bool spirv_translate_fp(const std::vector<FpInsn>& prog, bool fp32_output, std::vector<uint32_t>& out,
                        std::string* error, bool alpha_test = false, uint32_t alpha_func = 0x207,
                        uint8_t alpha_ref = 0, uint16_t tex_3d_mask = 0, uint16_t tex_shadow_mask = 0);

}  // namespace rsx
