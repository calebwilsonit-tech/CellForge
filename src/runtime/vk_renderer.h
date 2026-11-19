#pragma once
// GPU backend for RSX draws (Vulkan).
//
// The CPU front end (prepare_draw) does vertex work; the GPU rasterises, runs the
// translated fragment program, depth-tests and blends. Render targets stay on the
// GPU and are written back to guest memory by flush() — which the RSX calls at
// sync points (flips, semaphores, CPU-side clears, render-to-texture reads).

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "rsx_draw.h"

namespace rsx {

class GpuRenderer {
 public:
  virtual ~GpuRenderer() = default;

  // Creates a renderer on the best GPU (discrete preferred). nullptr if Vulkan is unavailable.
  static std::unique_ptr<GpuRenderer> create(std::function<void(std::string_view)> log);

  // Records the draw (CPU-prepared triangles). Returns false (with a reason) if the GPU path
  // can't handle it; the caller then renders it on the CPU after flush() + invalidate().
  virtual bool draw(const DrawEnv& env, const PreparedDraw& d, std::string* error) = 0;

  // Fast GPU vertex-shader path: translates RSX VP to SPIR-V, packs unique vertex inputs
  // and triangle indices directly into mapped GPU rings, and executes vertex shading,
  // clipping and culling natively on the GPU. Returns false to fall back to CPU prepare_draw.
  virtual bool draw_raw(const DrawEnv& env, const DrawCall& call, const PreparedDraw& d,
                        uint32_t& out_tris, std::string* error) = 0;

  // Clears colour and/or depth on the GPU. Returns false if the clear must fall back to the CPU.
  virtual bool clear(const DrawEnv& env, uint32_t mask) = 0;

  // Submits pending work and writes GPU-modified surfaces back to guest memory.
  virtual void flush(uint8_t* mem, bool include_depth = true) = 0;
  // Submits pending work and writes back only the colour surfaces overlapping
  // [addr, addr + size) (e.g. the buffer being displayed).
  virtual void flush_range(uint8_t* mem, uint32_t addr, uint32_t size) = 0;
  // Ends the frame: submits all pending GPU work without waiting for it. When the
  // GPU has finished, the colour surfaces overlapping [addr, addr + size) are
  // written to guest memory and `done` runs (on a later renderer call, at most
  // one frame later, or at the next synchronous flush).
  virtual void flip_async(uint8_t* mem, uint32_t addr, uint32_t size, std::function<void()> done) = 0;
  // Completes a submitted frame if the GPU has finished it (never blocks).
  virtual void poll() = 0;

  // The CPU changed guest memory in [addr, addr + size): surfaces overlapping it are
  // reloaded from guest memory before their next GPU use.
  virtual void invalidate(uint32_t addr, uint32_t size) = 0;

  virtual std::string device_name() const = 0;
};

}  // namespace rsx
