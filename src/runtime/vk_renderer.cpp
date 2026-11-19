#define _CRT_SECURE_NO_WARNINGS
#include "vk_renderer.h"

#include <set>
#include <string>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "profiler.h"
#include "spirv_gen.h"

namespace rsx {
namespace {

#define RVK_INSTANCE(X)                                                                                  \
  X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                    \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties)                     \
  X(vkGetPhysicalDeviceFeatures) X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define RVK_DEVICE(X)                                                                                    \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkQueueSubmit) X(vkDeviceWaitIdle) X(vkCreateCommandPool)     \
  X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)      \
  X(vkResetCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer)    \
  X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments)                        \
  X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)          \
  X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdSetViewport) X(vkCmdSetScissor)                \
  X(vkCmdSetBlendConstants) X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCreateBuffer)                           \
  X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory)               \
  X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkCreateImage) X(vkDestroyImage)              \
  X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView)       \
  X(vkCreateSampler) X(vkDestroySampler) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences)          \
  X(vkResetFences) X(vkCreateRenderPass) X(vkDestroyRenderPass) X(vkCreateFramebuffer)                  \
  X(vkDestroyFramebuffer) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout)    \
  X(vkDestroyPipelineLayout) X(vkCreateGraphicsPipelines) X(vkDestroyPipeline)                          \
  X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool)              \
  X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)  \
  X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp)                \
  X(vkGetQueryPoolResults) X(vkCmdCopyImage) X(vkGetFenceStatus)

struct Fns {
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
  PFN_vkCreateInstance vkCreateInstance = nullptr;
#define DECL(f) PFN_##f f = nullptr;
  RVK_INSTANCE(DECL) RVK_DEVICE(DECL)
#undef DECL
};

constexpr uint32_t kVertexFloats = 14 * 4;  // position + 13 varyings
constexpr VkDeviceSize kUploadRing = 96ull << 20, kVertexRing = 96ull << 20, kIndexRing = 32ull << 20,
                       kConstRing = 64ull << 20, kReadbackRing = 64ull << 20;
constexpr VkDeviceSize kVpUboSize = (2 + 512) * 16;  // vp_scale, vp_offset, c[512]
constexpr uint32_t kMaxSetsPerFlush = 8192;

uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }
uint16_t be16(const uint8_t* p) { return uint16_t((p[0] << 8) | p[1]); }
void put_be32(uint8_t* p, uint32_t v) { p[0] = uint8_t(v >> 24); p[1] = uint8_t(v >> 16); p[2] = uint8_t(v >> 8); p[3] = uint8_t(v); }
void put_be16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v >> 8); p[1] = uint8_t(v); }

VkBlendFactor blend_factor(uint32_t f) {
  switch (f) {
    case 0: return VK_BLEND_FACTOR_ZERO;
    case 1: return VK_BLEND_FACTOR_ONE;
    case 0x300: return VK_BLEND_FACTOR_SRC_COLOR;
    case 0x301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
    case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 0x306: return VK_BLEND_FACTOR_DST_COLOR;
    case 0x307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 0x308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
  }
}
VkBlendOp blend_op(uint32_t e) {
  switch (e) {
    case 0x8007: return VK_BLEND_OP_MIN;
    case 0x8008: return VK_BLEND_OP_MAX;
    case 0x800A: return VK_BLEND_OP_SUBTRACT;
    case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
    default: return VK_BLEND_OP_ADD;
  }
}
VkSamplerAddressMode address_mode(uint32_t w) {
  switch (w) {
    case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case 4: case 7: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  }
}

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  uint8_t* ptr = nullptr;
  VkDeviceSize size = 0, used = 0;
};

struct Image {
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkImageView srgb_view = VK_NULL_HANDLE;
  uint32_t w = 0, h = 0, d = 1;
};

struct Surface {
  Image image;
  uint32_t addr = 0, pitch = 0;
  bool depth = false, argb = true;
  uint32_t depth_fmt = 2;
  bool dirty = false, needs_upload = true;
  bool guarded = false;  // guest pages access-protected until the GPU data is written back
  uint64_t write_seq = 0;
  uint32_t span() const { return pitch * image.h; }
};

// ---- lazy readback ---------------------------------------------------------
// Render targets the GPU has drawn to are only copied back to guest memory when
// something needs them. Their guest pages are access-protected while the copy
// in guest memory is stale; the first CPU/SPU/DMA access faults, the handler
// reads the surface back (just that one), unprotects and the access retries.
// PS3RECOMP_NO_LAZY_READBACK disables this (stale render targets stay stale).
std::atomic<uint8_t*> g_guard_base{nullptr};
std::atomic<uint64_t> g_guard_pages[(uint64_t(1) << 32) / 4096 / 64];  // one bit per guest page
thread_local int t_in_renderer = 0;  // >0 while this thread runs renderer code that reads guest memory
struct InRenderer {
  InRenderer() { ++t_in_renderer; }
  ~InRenderer() { --t_in_renderer; }
};
bool guards_enabled() {
  static const bool on = std::getenv("PS3RECOMP_NO_LAZY_READBACK") == nullptr;
  return on;
}
bool page_guarded(uint32_t page) { return (g_guard_pages[page >> 6].load(std::memory_order_acquire) >> (page & 63)) & 1; }
void set_page_bits(uint32_t first, uint32_t last, bool on) {
  for (uint32_t p = first; p <= last; ++p) {
    const uint64_t bit = uint64_t(1) << (p & 63);
    if (on) g_guard_pages[p >> 6].fetch_or(bit, std::memory_order_acq_rel);
    else g_guard_pages[p >> 6].fetch_and(~bit, std::memory_order_acq_rel);
  }
}
class VulkanRenderer;
std::atomic<VulkanRenderer*> g_guard_renderer{nullptr};

struct Texture {
  Image image;
  uint64_t key = 0;
  uint64_t last_flush = 0;
  bool dynamic_rt = false;
};

class VulkanRenderer final : public GpuRenderer {
 public:
  explicit VulkanRenderer(std::function<void(std::string_view)> log) : log_(std::move(log)) {}
  ~VulkanRenderer() override { shutdown(); }

  bool init();
  bool draw(const DrawEnv& env, const PreparedDraw& d, std::string* error) override;
  bool draw_raw(const DrawEnv& env, const DrawCall& call, const PreparedDraw& d,
                uint32_t& out_tris, std::string* error) override;
  bool clear(const DrawEnv& env, uint32_t mask) override;
  void flush(uint8_t* mem, bool include_depth = true) override { flush_for(mem, include_depth, kFlushExternal); }
  void flush_range(uint8_t* mem, uint32_t addr, uint32_t size) override {
    if (!size) return;
    std::lock_guard<std::recursive_mutex> lk(m_);
    const uint64_t lo = addr, hi = uint64_t(addr) + size;
    if (other_.in_flight) {
      for (const auto& [s, off] : other_.pending) {
        if (s->addr < hi && uint64_t(s->addr) + s->span() > lo) { drain(); break; }
      }
    }
    for (const auto& s : surfaces_) {
      if (s->dirty && s->addr < hi && uint64_t(s->addr) + s->span() > lo) {
        flush_for(mem, true, kFlushExternal, lo, hi);
        return;
      }
    }
  }
  // Why a flush happened (recorded by the profiler as call ids 10020 + reason).
  enum FlushReason { kFlushExternal, kFlushResize, kFlushSurfaceLimit, kFlushTextureOverlap, kFlushRings, kFlushUpload };
  // [lo, hi): only surfaces overlapping this guest range are written back
  // (everything when PS3RECOMP_FULL_SYNC is set).
  void flush_for(uint8_t* mem, bool include_depth, FlushReason why, uint64_t lo = 0, uint64_t hi = ~0ull);
  void flush_impl(uint8_t* mem, bool include_depth, uint64_t lo, uint64_t hi);
  uint64_t draws_since_flush_ = 0;
  // Textures assembled on the GPU from render targets that share their memory
  // (e.g. two targets side by side in one pitch-linear buffer), so sampling them
  // needs no readback. Keyed by (address, width, height, pitch).
  bool composite_sources(const TexUnit& t, uint32_t target_addr, std::vector<Surface*>& out);
  VkImageView composite(const TexUnit& t, const std::vector<Surface*>& sources, std::string* error);
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, Image> composites_;
  // Open render pass shared by consecutive draws to the same targets.
  void end_pass();
  bool pass_open_ = false;
  VkFramebuffer pass_fb_ = VK_NULL_HANDLE;
  VkRenderPass pass_rp_ = VK_NULL_HANDLE;
  uint32_t pass_w_ = 0, pass_h_ = 0;
  VkQueryPool ts_pool_ = VK_NULL_HANDLE;
  // Per-draw GPU timing (profiling): queries 3.. hold a timestamp after each draw.
  static constexpr uint32_t kTsQueries = 3 + 1024;  // two per draw: before and after
  struct TsDraw { uint32_t fp; uint32_t tris; uint32_t w, h; };
  std::vector<TsDraw> ts_draws_;
  float ts_period_ns_ = 1.0f;
  bool ts_valid_ = false;
  void invalidate(uint32_t addr, uint32_t size) override;
  std::string device_name() const override { return name_; }

 private:
  void logf(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (log_) log_(b);
  }
  void shutdown();
  uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) const;
  bool make_buffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage);
  void free_buffer(Buffer& b);
  bool make_image(Image& im, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                  uint32_t d = 1, bool is_3d = false, bool mutable_srgb = false);
  void free_image(Image& im);
  void begin();
  void barrier_all();
  void transition_general(VkImage img, VkImageAspectFlags aspect);
  // Allocates from a ring; returns false if the ring is full (caller flushes).
  bool ring_alloc(Buffer& ring, VkDeviceSize size, VkDeviceSize& offset);

  Surface* surface(uint8_t* mem, uint32_t addr, uint32_t pitch, uint32_t w, uint32_t h, bool depth, uint32_t depth_fmt, bool argb);
  bool upload_surface(uint8_t* mem, Surface& s);
  void destroy_surface(Surface& s);
  VkImageView texture(const TexUnit& t, std::string* error);
  VkSampler sampler(const TexUnit& t);
  struct CachedVp {
    VkShaderModule mod = VK_NULL_HANDLE;
    SpirvVpMeta meta;
    uint64_t hash = 0;
    bool ok = false;
  };
  const CachedVp* get_vp(const VpState& vp, std::string* error);
  VkFramebuffer framebuffer(Surface* color, Surface* depth);
  VkPipeline pipeline(const PreparedDraw& d, bool with_depth, const CachedVp* vp, std::string* error);
  VkShaderModule shader(const std::vector<uint32_t>& code);

  std::function<void(std::string_view)> log_;
  std::recursive_mutex m_;
#if defined(_WIN32)
  HMODULE lib_ = nullptr;
#endif
  Fns vk_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  std::string name_;
  bool depth_clamp_ = false;

  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool recording_ = false;

  Buffer upload_, vertex_, index_, const_buf_, readback_;
  VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
  uint32_t sets_used_ = 0;
  VkRenderPass rp_color_ = VK_NULL_HANDLE, rp_depth_ = VK_NULL_HANDLE, rp_r16g16_ = VK_NULL_HANDLE;
  VkShaderModule vs_ = VK_NULL_HANDLE;
  VkShaderModule fs_d24_argb_ = VK_NULL_HANDLE, fs_abgr_y16x16_ = VK_NULL_HANDLE;
  VkPipeline pipe_d24_argb_ = VK_NULL_HANDLE, pipe_abgr_y16x16_ = VK_NULL_HANDLE;
  Image dummy_;
  VkSampler dummy_sampler_ = VK_NULL_HANDLE;

  struct ConvertedImage {
    Image image;
    VkFramebuffer fb = VK_NULL_HANDLE;
    uint64_t write_seq = ~0ull;
  };
  // GPU-reinterpreted render targets (keyed by addr, w, h, tag: 0x85=d24->argb, 0x95=abgr->y16x16, 0xFF=self-copy).
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, ConvertedImage> converted_;
  bool can_gpu_tex(const TexUnit& t, uint32_t target_color, uint32_t target_depth) const;
  VkImageView resolve_gpu_tex(const TexUnit& t, Surface* active_color, Surface* active_depth, std::string* error);
  VkPipeline make_convert_pipeline(VkShaderModule fs, VkRenderPass rp);
  bool run_convert_pass(Surface& src, ConvertedImage& dst, VkPipeline pipe, VkRenderPass rp);

  std::vector<std::unique_ptr<Surface>> surfaces_;
  std::map<std::pair<VkImageView, VkImageView>, VkFramebuffer> framebuffers_;
  std::unordered_map<uint64_t, VkPipeline> pipelines_;
  std::unordered_map<uint64_t, VkShaderModule> fragment_shaders_;
  std::unordered_map<uint64_t, CachedVp> vertex_shaders_;
  const VpState* last_vp_ptr_ = nullptr;
  uint64_t last_vp_prog_ver_ = 0;
  uint32_t last_vp_start_ = ~0u;
  const CachedVp* last_vp_ = nullptr;
  const VpState* last_vp_ubo_ptr_ = nullptr;
  uint64_t last_vp_ubo_ver_ = 0;
  float last_vp_s_[4]{}, last_vp_o_[4]{};
  VkDeviceSize last_vp_ubo_off_ = 0;
  bool last_vp_ubo_valid_ = false;
  std::unordered_map<uint64_t, Texture> textures_;
  uint64_t flush_seq_ = 1;
  using SamplerKey = std::tuple<bool, bool, uint32_t, uint32_t, uint32_t, bool, uint32_t, uint32_t>;
  std::map<SamplerKey, VkSampler> samplers_;
  std::vector<std::pair<Surface*, VkDeviceSize>> pending_readback_;

  // ---- two frames in flight ----
  // The members above (cmd_, fence_, rings, descriptor pool, ...) are the frame
  // being recorded. A flip submits it without waiting and swaps in the other
  // frame slot; the submitted frame is completed (fence wait, readback into
  // guest memory, completion callback such as presenting) when its slot is
  // needed again or before any synchronous flush.
  struct FrameState {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buffer upload, vertex, index, const_buf, readback;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    uint32_t sets_used = 0;
    VkQueryPool ts_pool = VK_NULL_HANDLE;
    std::vector<TsDraw> ts_draws;
    std::vector<std::pair<Surface*, VkDeviceSize>> pending;
    uint64_t draws = 0;
    bool in_flight = false;
    uint8_t* mem = nullptr;
    std::function<void()> done;
    std::chrono::steady_clock::time_point t_submit;
  };
  FrameState other_;
  void swap_frame();
  void complete(FrameState& f);  // finishes an in-flight frame
  void drain() { if (other_.in_flight) complete(other_); }
  void write_back(uint8_t* mem, const Buffer& readback, const std::vector<std::pair<Surface*, VkDeviceSize>>& pending);
  void record_readbacks(uint64_t lo, uint64_t hi, bool include_depth);
  void evict_textures();
  bool make_frame_resources(VkCommandBuffer& cmd, VkFence& fence, Buffer& upload, Buffer& vertex, Buffer& index,
                            Buffer& const_buf, Buffer& readback, VkDescriptorPool& pool, VkQueryPool& ts);

  // Lazy readback (see g_guard_pages).
  void mark_dirty(Surface* s);
  void guard_surface(Surface& s, bool on);

 public:
  // Called from the access-violation handler on any thread.
  void on_guard_fault(uint32_t guest_addr, bool write);
  void flip_async(uint8_t* mem, uint32_t addr, uint32_t size, std::function<void()> done) override;
  void poll() override {
    std::unique_lock<std::recursive_mutex> lk(m_, std::try_to_lock);
    if (lk.owns_lock() && other_.in_flight && vk_.vkGetFenceStatus(device_, other_.fence) == VK_SUCCESS) complete(other_);
  }
};

// ============================================================================
// setup
// ============================================================================
LONG CALLBACK guard_fault_handler(EXCEPTION_POINTERS* ep);

bool VulkanRenderer::init() {
#if defined(_WIN32)
  lib_ = LoadLibraryW(L"vulkan-1.dll");
  if (!lib_) return false;
  if (guards_enabled()) {
    g_guard_renderer = this;
    static const PVOID handler = AddVectoredExceptionHandler(1, guard_fault_handler);
    (void)handler;
  }
  vk_.vkGetInstanceProcAddr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(lib_, "vkGetInstanceProcAddr")));
#endif
  if (!vk_.vkGetInstanceProcAddr) return false;
  vk_.vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vk_.vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "ps3recomp-rsx";
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (!vk_.vkCreateInstance || vk_.vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) return false;
#define LOAD_I(f) vk_.f = reinterpret_cast<PFN_##f>(vk_.vkGetInstanceProcAddr(instance_, #f));
  RVK_INSTANCE(LOAD_I)
#undef LOAD_I

  uint32_t n = 0;
  vk_.vkEnumeratePhysicalDevices(instance_, &n, nullptr);
  std::vector<VkPhysicalDevice> gpus(n);
  vk_.vkEnumeratePhysicalDevices(instance_, &n, gpus.data());
  int best = -1;
  for (VkPhysicalDevice pd : gpus) {
    VkPhysicalDeviceProperties props;
    vk_.vkGetPhysicalDeviceProperties(pd, &props);
    uint32_t qn = 0;
    vk_.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vk_.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    for (uint32_t i = 0; i < qn; ++i) {
      if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
      const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2
                        : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
      if (score > best) {
        best = score; gpu_ = pd; queue_family_ = i; name_ = props.deviceName;
        ts_period_ns_ = props.limits.timestampPeriod;
        ts_valid_ = qf[i].timestampValidBits != 0;
      }
      break;
    }
  }
  if (!gpu_) return false;

  VkPhysicalDeviceFeatures have{};
  vk_.vkGetPhysicalDeviceFeatures(gpu_, &have);
  VkPhysicalDeviceFeatures want{};
  depth_clamp_ = have.depthClamp == VK_TRUE;
  want.depthClamp = have.depthClamp;
  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.pEnabledFeatures = &want;
  if (vk_.vkCreateDevice(gpu_, &dci, nullptr, &device_) != VK_SUCCESS) return false;
#define LOAD_D(f) vk_.f = reinterpret_cast<PFN_##f>(vk_.vkGetDeviceProcAddr(device_, #f));
  RVK_DEVICE(LOAD_D)
#undef LOAD_D
  vk_.vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = queue_family_;
  vk_.vkCreateCommandPool(device_, &pci, nullptr, &pool_);
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool_;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  vk_.vkAllocateCommandBuffers(device_, &cai, &cmd_);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vk_.vkCreateFence(device_, &fci, nullptr, &fence_);
  if (ts_valid_) {  // GPU execution time per submitted batch (profiling)
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = kTsQueries;
    if (vk_.vkCreateQueryPool(device_, &qpi, nullptr, &ts_pool_) != VK_SUCCESS) ts_pool_ = VK_NULL_HANDLE;
  }

  if (!make_buffer(upload_, kUploadRing, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
      !make_buffer(vertex_, kVertexRing, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
      !make_buffer(index_, kIndexRing, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ||
      !make_buffer(const_buf_, kConstRing, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
      !make_buffer(readback_, kReadbackRing, VK_BUFFER_USAGE_TRANSFER_DST_BIT))
    return false;

  // 16 combined image samplers (bindings 0..15) + 1 UBO for FP inline constants (binding 16)
  // + 1 UBO for VP viewport & 512 constants (binding 17).
  VkDescriptorSetLayoutBinding binds[18];
  for (uint32_t i = 0; i < 16; ++i)
    binds[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  binds[16] = {16, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  binds[17] = {17, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 18;
  dl.pBindings = binds;
  vk_.vkCreateDescriptorSetLayout(device_, &dl, nullptr, &set_layout_);
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &set_layout_;
  vk_.vkCreatePipelineLayout(device_, &pl, nullptr, &pipeline_layout_);
  VkDescriptorPoolSize ps[2] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxSetsPerFlush * 16},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxSetsPerFlush * 2},
  };
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = kMaxSetsPerFlush;
  dp.poolSizeCount = 2;
  dp.pPoolSizes = ps;
  vk_.vkCreateDescriptorPool(device_, &dp, nullptr, &desc_pool_);
  if (!make_frame_resources(other_.cmd, other_.fence, other_.upload, other_.vertex, other_.index, other_.const_buf,
                            other_.readback, other_.desc_pool, other_.ts_pool))
    return false;

  // Render passes: colour only, colour + depth, and R16G16 (for Y16_X16 GPU conversion).
  for (int with_depth = 0; with_depth < 2; ++with_depth) {
    VkAttachmentDescription att[2]{};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = att[0].finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    att[1] = att[0];
    att[1].format = VK_FORMAT_D32_SFLOAT;
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_GENERAL}, dref{1, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &cref;
    sub.pDepthStencilAttachment = with_depth ? &dref : nullptr;
    VkSubpassDependency deps[2]{};
    deps[0] = {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0};
    deps[1] = {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0};
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = with_depth ? 2 : 1;
    rp.pAttachments = att;
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;
    vk_.vkCreateRenderPass(device_, &rp, nullptr, with_depth ? &rp_depth_ : &rp_color_);
  }
  {
    VkAttachmentDescription att{};
    att.format = VK_FORMAT_R16G16_UNORM;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = att.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &cref;
    VkSubpassDependency deps[2]{};
    deps[0] = {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0};
    deps[1] = {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0};
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rp.attachmentCount = 1;
    rp.pAttachments = &att;
    rp.subpassCount = 1;
    rp.pSubpasses = &sub;
    rp.dependencyCount = 2;
    rp.pDependencies = deps;
    vk_.vkCreateRenderPass(device_, &rp, nullptr, &rp_r16g16_);
  }

  vs_ = shader(spirv_passthrough_vs());
  if (!vs_) return false;
  fs_d24_argb_ = shader(spirv_convert_d24_to_argb_fs());
  fs_abgr_y16x16_ = shader(spirv_convert_abgr_to_y16x16_fs());
  if (fs_d24_argb_) pipe_d24_argb_ = make_convert_pipeline(fs_d24_argb_, rp_color_);
  if (fs_abgr_y16x16_ && rp_r16g16_) pipe_abgr_y16x16_ = make_convert_pipeline(fs_abgr_y16x16_, rp_r16g16_);

  // A 1x1 texture for unused texture units.
  if (!make_image(dummy_, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT))
    return false;
  begin();
  transition_general(dummy_.img, VK_IMAGE_ASPECT_COLOR_BIT);
  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.maxLod = 0.0f;
  vk_.vkCreateSampler(device_, &sci, nullptr, &dummy_sampler_);
  logf("rsx: GPU renderer on %s", name_.c_str());
  return true;
}

bool VulkanRenderer::make_frame_resources(VkCommandBuffer& cmd, VkFence& fence, Buffer& upload, Buffer& vertex,
                                          Buffer& index, Buffer& const_buf, Buffer& readback, VkDescriptorPool& pool,
                                          VkQueryPool& ts) {
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool_;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  if (vk_.vkAllocateCommandBuffers(device_, &cai, &cmd) != VK_SUCCESS) return false;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vk_.vkCreateFence(device_, &fci, nullptr, &fence) != VK_SUCCESS) return false;
  if (ts_valid_) {
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = kTsQueries;
    if (vk_.vkCreateQueryPool(device_, &qpi, nullptr, &ts) != VK_SUCCESS) ts = VK_NULL_HANDLE;
  }
  if (!make_buffer(upload, kUploadRing, VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
      !make_buffer(vertex, kVertexRing, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
      !make_buffer(index, kIndexRing, VK_BUFFER_USAGE_INDEX_BUFFER_BIT) ||
      !make_buffer(const_buf, kConstRing, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
      !make_buffer(readback, kReadbackRing, VK_BUFFER_USAGE_TRANSFER_DST_BIT))
    return false;
  VkDescriptorPoolSize ps[2] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxSetsPerFlush * 16},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxSetsPerFlush * 2},
  };
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = kMaxSetsPerFlush;
  dp.poolSizeCount = 2;
  dp.pPoolSizes = ps;
  return vk_.vkCreateDescriptorPool(device_, &dp, nullptr, &pool) == VK_SUCCESS;
}

void VulkanRenderer::shutdown() {
  if (g_guard_renderer.load() == this) {
    g_guard_renderer = nullptr;
    for (auto& s : surfaces_)
      if (s->guarded) guard_surface(*s, false);
  }
  if (device_) {
    vk_.vkDeviceWaitIdle(device_);
    if (other_.desc_pool) vk_.vkDestroyDescriptorPool(device_, other_.desc_pool, nullptr);
    free_buffer(other_.upload);
    free_buffer(other_.vertex);
    free_buffer(other_.index);
    free_buffer(other_.const_buf);
    free_buffer(other_.readback);
    if (other_.fence) vk_.vkDestroyFence(device_, other_.fence, nullptr);
    if (other_.ts_pool) vk_.vkDestroyQueryPool(device_, other_.ts_pool, nullptr);
    if (ts_pool_) vk_.vkDestroyQueryPool(device_, ts_pool_, nullptr);
    for (auto& s : surfaces_) free_image(s->image);
    surfaces_.clear();
    for (auto& [k, c] : converted_) {
      if (c.fb) vk_.vkDestroyFramebuffer(device_, c.fb, nullptr);
      free_image(c.image);
    }
    converted_.clear();
    for (auto& [k, fb] : framebuffers_) vk_.vkDestroyFramebuffer(device_, fb, nullptr);
    for (auto& [k, p] : pipelines_) vk_.vkDestroyPipeline(device_, p, nullptr);
    if (pipe_d24_argb_) vk_.vkDestroyPipeline(device_, pipe_d24_argb_, nullptr);
    if (pipe_abgr_y16x16_) vk_.vkDestroyPipeline(device_, pipe_abgr_y16x16_, nullptr);
    if (fs_d24_argb_) vk_.vkDestroyShaderModule(device_, fs_d24_argb_, nullptr);
    if (fs_abgr_y16x16_) vk_.vkDestroyShaderModule(device_, fs_abgr_y16x16_, nullptr);
    for (auto& [k, s] : fragment_shaders_) vk_.vkDestroyShaderModule(device_, s, nullptr);
    for (auto& [k, v] : vertex_shaders_)
      if (v.mod) vk_.vkDestroyShaderModule(device_, v.mod, nullptr);
    for (auto& [k, t] : textures_) free_image(t.image);
    for (auto& [k, img] : composites_) free_image(img);
    for (auto& [k, s] : samplers_) vk_.vkDestroySampler(device_, s, nullptr);
    free_image(dummy_);
    if (dummy_sampler_) vk_.vkDestroySampler(device_, dummy_sampler_, nullptr);
    if (vs_) vk_.vkDestroyShaderModule(device_, vs_, nullptr);
    if (rp_color_) vk_.vkDestroyRenderPass(device_, rp_color_, nullptr);
    if (rp_depth_) vk_.vkDestroyRenderPass(device_, rp_depth_, nullptr);
    if (rp_r16g16_) vk_.vkDestroyRenderPass(device_, rp_r16g16_, nullptr);
    if (desc_pool_) vk_.vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
    if (pipeline_layout_) vk_.vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    free_buffer(upload_);
    free_buffer(vertex_);
    free_buffer(index_);
    free_buffer(const_buf_);
    free_buffer(readback_);
    if (fence_) vk_.vkDestroyFence(device_, fence_, nullptr);
    if (pool_) vk_.vkDestroyCommandPool(device_, pool_, nullptr);
    vk_.vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }
  if (instance_) {
    vk_.vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
#if defined(_WIN32)
  if (lib_) FreeLibrary(lib_);
  lib_ = nullptr;
#endif
}

uint32_t VulkanRenderer::memory_type(uint32_t bits, VkMemoryPropertyFlags want) const {
  VkPhysicalDeviceMemoryProperties mp;
  vk_.vkGetPhysicalDeviceMemoryProperties(gpu_, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}

bool VulkanRenderer::make_buffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = usage;
  if (vk_.vkCreateBuffer(device_, &bci, nullptr, &b.buf) != VK_SUCCESS) return false;
  VkMemoryRequirements mr;
  vk_.vkGetBufferMemoryRequirements(device_, b.buf, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  // Buffers the CPU reads back must be host-cached: write-combined memory is fast to
  // write but extremely slow to read (it made every flip take ~300 ms).
  const VkMemoryPropertyFlags base = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  mai.memoryTypeIndex = UINT32_MAX;
  if (usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT)
    mai.memoryTypeIndex = memory_type(mr.memoryTypeBits, base | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = memory_type(mr.memoryTypeBits, base);
  if (mai.memoryTypeIndex == UINT32_MAX || vk_.vkAllocateMemory(device_, &mai, nullptr, &b.mem) != VK_SUCCESS) return false;
  vk_.vkBindBufferMemory(device_, b.buf, b.mem, 0);
  void* p = nullptr;
  vk_.vkMapMemory(device_, b.mem, 0, size, 0, &p);
  b.ptr = static_cast<uint8_t*>(p);
  b.size = size;
  b.used = 0;
  return true;
}

void VulkanRenderer::free_buffer(Buffer& b) {
  if (b.ptr) vk_.vkUnmapMemory(device_, b.mem);
  if (b.buf) vk_.vkDestroyBuffer(device_, b.buf, nullptr);
  if (b.mem) vk_.vkFreeMemory(device_, b.mem, nullptr);
  b = Buffer{};
}

bool VulkanRenderer::make_image(Image& im, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                                VkImageAspectFlags aspect, uint32_t d, bool is_3d, bool mutable_srgb) {
  const bool make_srgb_view = mutable_srgb && fmt == VK_FORMAT_R8G8B8A8_UNORM;
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.flags = make_srgb_view ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
  ici.imageType = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  ici.format = fmt;
  ici.extent = {w, h, is_3d ? std::max(1u, d) : 1u};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = usage;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vk_.vkCreateImage(device_, &ici, nullptr, &im.img) != VK_SUCCESS) return false;
  VkMemoryRequirements mr;
  vk_.vkGetImageMemoryRequirements(device_, im.img, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (mai.memoryTypeIndex == UINT32_MAX || vk_.vkAllocateMemory(device_, &mai, nullptr, &im.mem) != VK_SUCCESS) return false;
  vk_.vkBindImageMemory(device_, im.img, im.mem, 0);
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = im.img;
  vci.viewType = is_3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
  vci.format = fmt;
  if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
    vci.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
  } else if (fmt == VK_FORMAT_R16G16_UNORM) {
    vci.components = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE};
  }
  vci.subresourceRange = {aspect, 0, 1, 0, 1};
  if (vk_.vkCreateImageView(device_, &vci, nullptr, &im.view) != VK_SUCCESS) return false;
  if (make_srgb_view) {
    vci.format = VK_FORMAT_R8G8B8A8_SRGB;
    if (vk_.vkCreateImageView(device_, &vci, nullptr, &im.srgb_view) != VK_SUCCESS) im.srgb_view = VK_NULL_HANDLE;
  }
  im.w = w;
  im.h = h;
  im.d = is_3d ? std::max(1u, d) : 1u;
  return true;
}

void VulkanRenderer::free_image(Image& im) {
  if (im.srgb_view) vk_.vkDestroyImageView(device_, im.srgb_view, nullptr);
  if (im.view) vk_.vkDestroyImageView(device_, im.view, nullptr);
  if (im.img) vk_.vkDestroyImage(device_, im.img, nullptr);
  if (im.mem) vk_.vkFreeMemory(device_, im.mem, nullptr);
  im = Image{};
}

void VulkanRenderer::begin() {
  if (recording_) return;
  vk_.vkResetCommandBuffer(cmd_, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_.vkBeginCommandBuffer(cmd_, &bi);
  recording_ = true;
  if (ts_pool_ && prof::enabled()) {
    vk_.vkCmdResetQueryPool(cmd_, ts_pool_, 0, kTsQueries);
    ts_draws_.clear();
    vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool_, 0);
  }
}

// Consecutive draws to the same colour/depth targets share one render pass (each
// pass loads and stores whole attachments, so a pass per draw is very slow). Any
// command that cannot run inside a pass (barriers, transitions, copies, clears)
// closes it first.
void VulkanRenderer::end_pass() {
  if (!pass_open_) return;
  pass_open_ = false;
  vk_.vkCmdEndRenderPass(cmd_);
  barrier_all();
}

void VulkanRenderer::barrier_all() {
  end_pass();
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk_.vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
}

void VulkanRenderer::transition_general(VkImage img, VkImageAspectFlags aspect) {
  end_pass();
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {aspect, 0, 1, 0, 1};
  vk_.vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &b);
}

bool VulkanRenderer::ring_alloc(Buffer& ring, VkDeviceSize size, VkDeviceSize& offset) {
  const VkDeviceSize aligned = (ring.used + 255) & ~VkDeviceSize(255);
  if (aligned + size > ring.size) return false;
  offset = aligned;
  ring.used = aligned + size;
  return true;
}

VkShaderModule VulkanRenderer::shader(const std::vector<uint32_t>& code) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = code.size() * 4;
  ci.pCode = code.data();
  VkShaderModule m = VK_NULL_HANDLE;
  if (vk_.vkCreateShaderModule(device_, &ci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
  return m;
}

// ============================================================================
// surfaces (render targets resident on the GPU)
// ============================================================================
Surface* VulkanRenderer::surface(uint8_t* mem, uint32_t addr, uint32_t pitch, uint32_t w, uint32_t h, bool depth,
                                 uint32_t depth_fmt, bool argb) {
  for (auto& s : surfaces_) {
    if (s->addr != addr || s->depth != depth) continue;
    if (s->pitch == pitch && s->image.w >= w && s->image.h >= h && s->argb == argb && s->depth_fmt == depth_fmt)
      return s.get();
  }
  for (auto& s : surfaces_) {
    if (s->addr != addr || s->depth != depth || s->pitch != pitch || s->argb != argb || s->depth_fmt != depth_fmt) continue;
    // Same surface with a larger clip bound than previously seen: expand to the max dimensions.
    w = std::max(w, s->image.w);
    h = std::max(h, s->image.h);
    if (s->dirty) flush_for(mem, s->depth, kFlushResize, s->addr, uint64_t(s->addr) + s->span());
    destroy_surface(*s);
    s = std::move(surfaces_.back());
    surfaces_.pop_back();
    break;
  }
  if (surfaces_.size() >= 64) {
    flush_for(mem, false, kFlushSurfaceLimit);
    for (auto& s : surfaces_) destroy_surface(*s);
    surfaces_.clear();
  }
  auto s = std::make_unique<Surface>();
  s->addr = addr;
  s->pitch = pitch;
  s->depth = depth;
  s->depth_fmt = depth_fmt;
  s->argb = argb;
  const bool ok = depth ? make_image(s->image, w, h, VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT)
                        : make_image(s->image, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT, 1, false, true);
  if (!ok) return nullptr;
  begin();
  transition_general(s->image.img, depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
  s->needs_upload = !depth;
  surfaces_.push_back(std::move(s));
  return surfaces_.back().get();
}

bool VulkanRenderer::upload_surface(uint8_t* mem, Surface& s) {
  if (s.guarded) guard_surface(s, false);  // guest memory is the source of truth here
  const uint32_t w = s.image.w, h = s.image.h;
  VkDeviceSize off = 0;
  if (!ring_alloc(upload_, VkDeviceSize(w) * h * 4, off)) return false;
  if (prof::enabled()) prof::record_call(10040, s.depth ? 1 : 0, uint64_t(w) * h * 4 / 1024, "gpu:surface-upload-KB");
  uint8_t* dst = upload_.ptr + off;
  if (s.depth) {
    for (uint32_t y = 0; y < h; ++y) {
      const uint8_t* row = mem + s.addr + size_t(y) * s.pitch;
      for (uint32_t x = 0; x < w; ++x) {
        float z;
        if (s.depth_fmt == 1) z = be16(row + 2 * x) / 65535.0f;
        else z = (be32(row + 4 * x) >> 8) / 16777215.0f;
        std::memcpy(dst + (size_t(y) * w + x) * 4, &z, 4);
      }
    }
  } else {
    const __m128i shuf = s.argb
        ? _mm_setr_epi8(1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12)
        : _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    for (uint32_t y = 0; y < h; ++y) {
      const uint8_t* row = mem + s.addr + size_t(y) * s.pitch;
      uint8_t* out_row = dst + size_t(y) * w * 4;
      uint32_t x = 0;
      for (; x + 4 <= w; x += 4) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 4 * x));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_row + 4 * x), _mm_shuffle_epi8(v, shuf));
      }
      for (; x < w; ++x) {
        const uint32_t v = be32(row + 4 * x);
        uint8_t* o = out_row + 4 * x;
        if (s.argb) { o[0] = uint8_t(v >> 16); o[1] = uint8_t(v >> 8); o[2] = uint8_t(v); }
        else { o[0] = uint8_t(v); o[1] = uint8_t(v >> 8); o[2] = uint8_t(v >> 16); }
        o[3] = uint8_t(v >> 24);
      }
    }
  }
  begin();
  barrier_all();
  VkBufferImageCopy c{};
  c.bufferOffset = off;
  c.imageSubresource = {VkImageAspectFlags(s.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
  c.imageExtent = {w, h, 1};
  vk_.vkCmdCopyBufferToImage(cmd_, upload_.buf, s.image.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
  barrier_all();
  s.needs_upload = false;
  ++s.write_seq;
  return true;
}

void VulkanRenderer::destroy_surface(Surface& s) {
  if (recording_) end_pass();
  drain();  // an in-flight frame may still read back into / reference this surface
  if (s.guarded) guard_surface(s, false);
  vk_.vkDeviceWaitIdle(device_);
  for (auto& [k, c] : converted_)
    if (std::get<0>(k) == s.addr) c.write_seq = ~0ull;
  for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
    if (it->first.first == s.image.view || it->first.second == s.image.view) {
      vk_.vkDestroyFramebuffer(device_, it->second, nullptr);
      it = framebuffers_.erase(it);
    } else {
      ++it;
    }
  }
  free_image(s.image);
}

void VulkanRenderer::invalidate(uint32_t addr, uint32_t size) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  for (auto& s : surfaces_)
    if (addr < s->addr + s->span() && s->addr < addr + size) {
      s->needs_upload = true;
      ++s->write_seq;
    }
}

VkFramebuffer VulkanRenderer::framebuffer(Surface* color, Surface* depth) {
  const auto key = std::make_pair(color->image.view, depth ? depth->image.view : VkImageView(VK_NULL_HANDLE));
  auto it = framebuffers_.find(key);
  if (it != framebuffers_.end()) return it->second;
  VkImageView views[2] = {color->image.view, depth ? depth->image.view : VK_NULL_HANDLE};
  VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  fci.renderPass = depth ? rp_depth_ : rp_color_;
  fci.attachmentCount = depth ? 2 : 1;
  fci.pAttachments = views;
  fci.width = depth ? std::min(color->image.w, depth->image.w) : color->image.w;
  fci.height = depth ? std::min(color->image.h, depth->image.h) : color->image.h;
  fci.layers = 1;
  VkFramebuffer fb = VK_NULL_HANDLE;
  vk_.vkCreateFramebuffer(device_, &fci, nullptr, &fb);
  framebuffers_[key] = fb;
  return fb;
}

// ============================================================================
// textures
// ============================================================================
namespace {
// Fast content hash: four independent 64-bit lanes (multiply-rotate mixing), so
// it runs at memory bandwidth instead of one byte per multiply.
uint64_t hash_bytes(const uint8_t* p, size_t n) {
  constexpr uint64_t k1 = 0x9E3779B97F4A7C15ull, k2 = 0xC2B2AE3D27D4EB4Full;
  uint64_t a = k1, b = k2, c = k1 ^ k2, d = ~k1;
  auto mix = [](uint64_t acc, uint64_t v) { return std::rotl(acc ^ (v * 0xC2B2AE3D27D4EB4Full), 31) * 0x9E3779B97F4A7C15ull; };
  auto mix_block32 = [&](const uint8_t* q) {
    uint64_t w[4];
    std::memcpy(w, q, 32);
    a = mix(a, w[0]); b = mix(b, w[1]); c = mix(c, w[2]); d = mix(d, w[3]);
  };
  if (n > 16384) {
    // For large textures (e.g. 720p video frames or mipmapped sheets), sample head, tail
    // and 256 evenly-spaced 32-byte blocks (~16 KB total) instead of scanning megabytes per draw.
    for (size_t i = 0; i < 4096; i += 32) mix_block32(p + i);
    const size_t span = n - 8192;
    const size_t step = std::max<size_t>(32, span / 256);
    for (size_t i = 4096; i + 32 <= n - 4096; i += step) mix_block32(p + i);
    for (size_t i = n - 4096; i + 32 <= n; i += 32) mix_block32(p + i);
  } else {
    size_t i = 0;
    for (; i + 32 <= n; i += 32) mix_block32(p + i);
    for (; i + 8 <= n; i += 8) { uint64_t w; std::memcpy(&w, p + i, 8); a = mix(a, w); }
    for (; i < n; ++i) b = mix(b, p[i]);
  }
  uint64_t h = std::rotl(a, 1) + std::rotl(b, 7) + std::rotl(c, 12) + std::rotl(d, 18) + n;
  h ^= h >> 33; h *= k2; h ^= h >> 29;
  return h;
}
}  // namespace

VkImageView VulkanRenderer::texture(const TexUnit& t, std::string* error) {
  // Key: location, shape, format and a hash of the bytes (catches CPU updates).
  const uint32_t size = t.byte_size();
  uint64_t h = hash_bytes(t.base, size);
  bool dynamic_rt = false;
  for (const auto& s : surfaces_) {
    if (s->addr && t.address < s->addr + s->span() && s->addr < t.address + size) {
      h ^= (s->write_seq + 1) * 0xBF58476D1CE4E5B9ull;
      dynamic_rt = true;
    }
  }
  const uint32_t d = t.is_3d() ? t.depth : 1u;
  h ^= (uint64_t(t.address) << 1) ^ (uint64_t(t.format) << 40) ^ (uint64_t(t.width) << 20) ^ (uint64_t(t.height) << 52) ^
       (uint64_t(d) * 0x517CC1B727220A95ull) ^ (uint64_t(t.dimension) << 60) ^
       (t.linear_layout ? 0x5555ull : 0ull) ^ (t.is_srgb() ? 0xAAAA0000ull : 0ull) ^
       (t.is_shadow() ? 0x333300000000ull : 0ull) ^ (uint64_t(t.pitch) * 0x9E3779B97F4A7C15ull);
  auto it = textures_.find(h);
  if (it != textures_.end()) {
    it->second.last_flush = flush_seq_;
    return it->second.image.view;
  }

  Texture tex;
  tex.key = h;
  tex.last_flush = flush_seq_;
  tex.dynamic_rt = dynamic_rt;
  if (t.is_shadow()) {
    // CPU-backed depth texture used with hardware shadow compare: upload as D32_SFLOAT.
    std::vector<float> zbuf(size_t(t.width) * t.height);
    const bool is16 = (t.format == 0x91 || t.format == 0x92);
    const uint32_t bpp = is16 ? 2u : 4u;
    const uint32_t stride = t.pitch ? t.pitch : t.width * bpp;
    const uint32_t lw = uint32_t(std::countr_zero(std::bit_ceil(t.width)));
    const uint32_t lh = uint32_t(std::countr_zero(std::bit_ceil(t.height)));
    auto morton2d = [&](uint32_t x, uint32_t y) {
      uint32_t off = 0, shift = 0;
      const uint32_t both = std::min(lw, lh);
      for (uint32_t i = 0; i < both; ++i) {
        off |= ((x >> i) & 1u) << shift++;
        off |= ((y >> i) & 1u) << shift++;
      }
      return off | ((lw > lh ? (x >> both) : (y >> both)) << shift);
    };
    for (uint32_t y = 0; y < t.height; ++y) {
      for (uint32_t x = 0; x < t.width; ++x) {
        const size_t pos = t.linear_layout ? (size_t(y) * stride + size_t(x) * bpp) : (size_t(morton2d(x, y)) * bpp);
        const uint8_t* p = t.base + pos;
        zbuf[size_t(y) * t.width + x] = is16 ? (be16(p) / 65535.0f) : ((be32(p) >> 8) / 16777215.0f);
      }
    }
    if (!make_image(tex.image, t.width, t.height, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_DEPTH_BIT)) {
      if (error) *error = "depth texture allocation failed";
      return VK_NULL_HANDLE;
    }
    VkDeviceSize off = 0;
    if (!ring_alloc(upload_, zbuf.size() * 4, off)) {
      free_image(tex.image);
      if (error) *error = "upload ring full";
      return VK_NULL_HANDLE;
    }
    std::memcpy(upload_.ptr + off, zbuf.data(), zbuf.size() * 4);
    begin();
    transition_general(tex.image.img, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkBufferImageCopy c{};
    c.bufferOffset = off;
    c.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    c.imageExtent = {t.width, t.height, 1};
    vk_.vkCmdCopyBufferToImage(cmd_, upload_.buf, tex.image.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    barrier_all();
    const VkImageView view = tex.image.view;
    textures_[h] = tex;
    return view;
  }

  std::vector<uint32_t> rgba;
  decode_texture_rgba(t, rgba);
  const VkFormat fmt = t.is_srgb() ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  if (!make_image(tex.image, t.width, t.height, fmt,
                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                  d, t.is_3d(), false)) {
    if (error) *error = "texture allocation failed";
    return VK_NULL_HANDLE;
  }
  VkDeviceSize off = 0;
  if (prof::enabled()) prof::record_call(10041, t.format, rgba.size() * 4 / 1024, "gpu:texture-upload-KB");
  if (!ring_alloc(upload_, rgba.size() * 4, off)) {
    free_image(tex.image);
    if (error) *error = "upload ring full";
    return VK_NULL_HANDLE;
  }
  std::memcpy(upload_.ptr + off, rgba.data(), rgba.size() * 4);
  begin();
  transition_general(tex.image.img, VK_IMAGE_ASPECT_COLOR_BIT);
  VkBufferImageCopy c{};
  c.bufferOffset = off;
  c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  c.imageExtent = {t.width, t.height, d};
  vk_.vkCmdCopyBufferToImage(cmd_, upload_.buf, tex.image.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
  barrier_all();
  const VkImageView view = tex.image.view;
  textures_[h] = tex;
  return view;
}

// A linear A8R8G8B8 texture overlapping GPU colour surfaces can be assembled on the
// GPU when every overlapping surface is an ARGB target with the texture's pitch,
// starts inside the texture, and is not the target currently being drawn.
bool VulkanRenderer::composite_sources(const TexUnit& t, uint32_t target_addr, std::vector<Surface*>& out) {
  out.clear();
  if (t.is_3d() || t.format != 0x85 || !t.linear_layout || !t.pitch) return false;
  const uint64_t lo = t.address, hi = lo + uint64_t(t.pitch) * t.height;
  for (auto& s : surfaces_) {
    if (!s->addr || !(s->addr < hi && uint64_t(s->addr) + s->span() > lo)) continue;
    if (s->depth) {
      if (s->dirty) return false;
      continue;
    }
    if (s->addr == target_addr || !s->argb || s->pitch != t.pitch || s->addr < lo) return false;
    out.push_back(s.get());
  }
  return !out.empty();
}

VkImageView VulkanRenderer::composite(const TexUnit& t, const std::vector<Surface*>& sources, std::string* error) {
  Image& img = composites_[{t.address, t.width, t.height, t.pitch}];
  begin();
  if (!img.img) {
    if (!make_image(img, t.width, t.height, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                    1, false, true)) {
      if (error) *error = "composite texture allocation failed";
      return VK_NULL_HANDLE;
    }
    transition_general(img.img, VK_IMAGE_ASPECT_COLOR_BIT);
  }
  // Placement of each surface inside the texture, and how much of it they cover.
  struct Place { Surface* s; uint32_t x, y, w, h; };
  std::vector<Place> places;
  uint64_t covered = 0;
  for (Surface* s : sources) {
    const uint32_t off = s->addr - t.address;
    const uint32_t x = (off % t.pitch) / 4, y = off / t.pitch;
    if (x >= t.width || y >= t.height) continue;
    const uint32_t w = std::min(s->image.w, t.width - x), h = std::min(s->image.h, t.height - y);
    if (s->needs_upload) continue;  // guest memory is authoritative for this one
    places.push_back({s, x, y, w, h});
    covered += uint64_t(w) * h;
  }
  if (covered < uint64_t(t.width) * t.height) {
    // Parts not covered by a GPU surface come from guest memory.
    std::vector<uint32_t> rgba;
    decode_texture_rgba(t, rgba);
    VkDeviceSize off = 0;
    if (!ring_alloc(upload_, rgba.size() * 4, off)) {
      flush_for(nullptr, false, kFlushUpload, 0, 0);
      if (!ring_alloc(upload_, rgba.size() * 4, off)) { if (error) *error = "upload ring full"; return VK_NULL_HANDLE; }
      begin();
    }
    std::memcpy(upload_.ptr + off, rgba.data(), rgba.size() * 4);
    barrier_all();
    VkBufferImageCopy c{};
    c.bufferOffset = off;
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageExtent = {t.width, t.height, 1};
    vk_.vkCmdCopyBufferToImage(cmd_, upload_.buf, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
  }
  barrier_all();
  for (const Place& p : places) {
    VkImageCopy r{};
    r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.dstOffset = {int32_t(p.x), int32_t(p.y), 0};
    r.extent = {p.w, p.h, 1};
    vk_.vkCmdCopyImage(cmd_, p.s->image.img, VK_IMAGE_LAYOUT_GENERAL, img.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
  }
  barrier_all();
  return (t.is_srgb() && img.srgb_view) ? img.srgb_view : img.view;
}

VkSampler VulkanRenderer::sampler(const TexUnit& t) {
  const bool shadow = t.is_shadow();
  const uint32_t wrap_r = t.is_3d() ? t.wrap_r : 3u;
  const uint32_t zfunc = shadow ? (t.zfunc & 7u) : 7u;
  uint32_t border_mode = 0;  // 0=transparent black, 1=opaque black, 2=opaque white
  if (shadow) {
    // Outside the shadow map frustum, depth 1.0 (white) avoids spurious shadows.
    border_mode = (t.border_color == 0u || (t.border_color & 0x00FFFFFFu) >= 0x00800000u) ? 2u : 1u;
  } else if (t.border_color == 0xFFFFFFFFu) {
    border_mode = 2u;
  } else if (t.border_color == 0xFF000000u) {
    border_mode = 1u;
  }
  const SamplerKey key{t.mag_linear, t.min_linear, t.wrap_s, t.wrap_t, wrap_r, shadow, zfunc, border_mode};
  auto it = samplers_.find(key);
  if (it != samplers_.end()) return it->second;
  // RSX compares D_tex <zfunc> D_ref; Vulkan compares D_ref <compareOp> D_tex.
  static constexpr VkCompareOp kShadowCompare[8] = {
      VK_COMPARE_OP_NEVER,            // 0: NEVER
      VK_COMPARE_OP_GREATER,          // 1: D_tex <  D_ref  <=>  D_ref >  D_tex
      VK_COMPARE_OP_EQUAL,            // 2: D_tex == D_ref  <=>  D_ref == D_tex
      VK_COMPARE_OP_GREATER_OR_EQUAL, // 3: D_tex <= D_ref  <=>  D_ref >= D_tex
      VK_COMPARE_OP_LESS,             // 4: D_tex >  D_ref  <=>  D_ref <  D_tex
      VK_COMPARE_OP_NOT_EQUAL,        // 5: D_tex != D_ref  <=>  D_ref != D_tex
      VK_COMPARE_OP_LESS_OR_EQUAL,    // 6: D_tex >= D_ref  <=>  D_ref <= D_tex
      VK_COMPARE_OP_ALWAYS,           // 7: ALWAYS
  };
  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = t.mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  sci.minFilter = t.min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  sci.addressModeU = address_mode(t.wrap_s);
  sci.addressModeV = address_mode(t.wrap_t);
  sci.addressModeW = t.is_3d() ? address_mode(wrap_r) : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.compareEnable = shadow ? VK_TRUE : VK_FALSE;
  sci.compareOp = shadow ? kShadowCompare[zfunc] : VK_COMPARE_OP_ALWAYS;
  sci.borderColor = border_mode == 2u ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                    : border_mode == 1u ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                        : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  sci.maxLod = 0.0f;
  VkSampler s = VK_NULL_HANDLE;
  vk_.vkCreateSampler(device_, &sci, nullptr, &s);
  samplers_[key] = s;
  return s;
}

// ============================================================================
// pipelines
// ============================================================================
const VulkanRenderer::CachedVp* VulkanRenderer::get_vp(const VpState& vp, std::string* error) {
  if (vp.prog_ver > 1 && last_vp_ && last_vp_ptr_ == &vp && last_vp_prog_ver_ == vp.prog_ver &&
      last_vp_start_ == vp.start) {
    if (!last_vp_->ok && error) *error = "vertex program has no GPU translation";
    return last_vp_;
  }
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
  bool found_end = false;
  for (uint32_t pc = vp.start; pc < vp.program.size() && pc - vp.start < 512; ++pc) {
    const VpInsn& i = vp.program[pc];
    mix((uint64_t(i.vec_op) << 0) | (uint64_t(i.sca_op) << 8) | (uint64_t(i.input) << 16) |
        (uint64_t(i.constant) << 24) | (uint64_t(i.dst_tmp) << 40) | (uint64_t(i.dst_out) << 48) |
        (uint64_t(i.sca_dst_tmp) << 56));
    mix((uint64_t(i.vec_mask) << 0) | (uint64_t(i.sca_mask) << 8) | (uint64_t(i.addr_reg) << 16) |
        (uint64_t(i.addr_swz) << 20) | (uint64_t(i.cond) << 24) | (uint64_t(i.cond_reg) << 28) |
        (uint64_t(i.vec_result ? 1 : 0) << 32) | (uint64_t(i.saturate ? 1 : 0) << 33) |
        (uint64_t(i.cond_test ? 1 : 0) << 34) | (uint64_t(i.end ? 1 : 0) << 35) |
        (uint64_t(i.index_const ? 1 : 0) << 36) | (uint64_t(i.cond_update ? 1 : 0) << 37) |
        (uint64_t(i.branch_target) << 40));
    mix((uint64_t(i.cond_swz[0]) << 0) | (uint64_t(i.cond_swz[1]) << 4) |
        (uint64_t(i.cond_swz[2]) << 8) | (uint64_t(i.cond_swz[3]) << 12));
    for (int k = 0; k < 3; ++k) {
      const VpSrc& s = i.src[k];
      mix((uint64_t(s.type) << 0) | (uint64_t(s.index) << 8) | (uint64_t(s.swz[0]) << 16) |
          (uint64_t(s.swz[1]) << 20) | (uint64_t(s.swz[2]) << 24) | (uint64_t(s.swz[3]) << 28) |
          (uint64_t(s.neg ? 1 : 0) << 32) | (uint64_t(s.abs ? 1 : 0) << 33));
    }
    if (i.end) { found_end = true; break; }
  }
  if (!found_end) {
    if (error) *error = "vertex program missing end marker";
    return nullptr;
  }
  auto it = vertex_shaders_.find(h);
  if (it == vertex_shaders_.end()) {
    CachedVp entry;
    entry.hash = h;
    std::vector<uint32_t> code;
    std::string verr;
    if (spirv_translate_vp(vp, code, entry.meta, &verr)) {
      entry.mod = shader(code);
      if (entry.mod) entry.ok = true;
      else if (error) *error = "the driver rejected a translated vertex program";
    } else {
      if (error) *error = verr;
    }
    it = vertex_shaders_.emplace(h, entry).first;
  } else if (!it->second.ok && error) {
    *error = "vertex program has no GPU translation";
  }
  if (vp.prog_ver > 1) {
    last_vp_ptr_ = &vp;
    last_vp_prog_ver_ = vp.prog_ver;
    last_vp_start_ = vp.start;
    last_vp_ = &it->second;
  }
  return &it->second;
}

VkPipeline VulkanRenderer::pipeline(const PreparedDraw& d, bool with_depth, const CachedVp* vp, std::string* error) {
  const bool use_alpha_test = d.alpha_test && d.alpha_func != 0x207;
  uint16_t tex_3d_mask = 0, tex_shadow_mask = 0;
  for (uint32_t u = 0; u < 16; ++u) {
    if (!d.tex[u].enabled || !d.tex[u].base) continue;
    if (d.tex[u].is_3d()) tex_3d_mask |= uint16_t(1u << u);
    else if (d.tex[u].is_shadow()) tex_shadow_mask |= uint16_t(1u << u);
  }
  uint64_t key = d.fp_hash;
  auto mix = [&](uint64_t v) { key = (key ^ v) * 1099511628211ull; };
  mix(d.blend); mix(d.sf_rgb); mix(d.sf_a); mix(d.df_rgb); mix(d.df_a); mix(d.eq_rgb); mix(d.eq_a);
  mix(d.color_mask); mix(with_depth); mix(d.depth_test); mix(d.depth_write); mix(d.depth_func);
  mix(use_alpha_test); mix(use_alpha_test ? d.alpha_func : 0x207u); mix(use_alpha_test ? d.alpha_ref : 0u);
  mix(tex_3d_mask); mix(tex_shadow_mask);
  mix(vp ? vp->hash : 0ull);
  mix(vp ? (1u + vp->meta.in_count) : 0u);
  mix(vp && d.cull ? 1u : 0u);
  mix(vp && d.cull ? d.cull_face : 0u);
  mix(vp && d.cull ? (d.front_ccw ? 1u : 0u) : 0u);
  auto it = pipelines_.find(key);
  if (it != pipelines_.end()) return it->second;

  uint64_t fkey = d.fp_hash;
  if (use_alpha_test) {
    fkey = (fkey ^ d.alpha_func ^ 0xA1FA7E57u) * 1099511628211ull;
    fkey = (fkey ^ d.alpha_ref) * 1099511628211ull;
  }
  if (tex_3d_mask || tex_shadow_mask) {
    fkey = (fkey ^ ((uint64_t(tex_3d_mask) << 16) | tex_shadow_mask) ^ 0x3D5AD000ull) * 1099511628211ull;
  }
  VkShaderModule fs = VK_NULL_HANDLE;
  auto fit = fragment_shaders_.find(fkey);
  if (fit != fragment_shaders_.end()) {
    fs = fit->second;
  } else {
    std::vector<uint32_t> code;
    if (!spirv_translate_fp(d.fp, d.fp32_out, code, error, use_alpha_test, d.alpha_func, d.alpha_ref,
                            tex_3d_mask, tex_shadow_mask))
      return VK_NULL_HANDLE;
    fs = shader(code);
    if (!fs) {
      if (error) *error = "the driver rejected a translated fragment program";
      return VK_NULL_HANDLE;
    }
    fragment_shaders_[fkey] = fs;
  }

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vp ? vp->mod : vs_;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";

  VkVertexInputBindingDescription bind{};
  VkVertexInputAttributeDescription attrs[16];
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  if (vp) {
    if (vp->meta.in_count > 0) {
      bind = {0, vp->meta.in_count * 16u, VK_VERTEX_INPUT_RATE_VERTEX};
      for (uint32_t i = 0; i < vp->meta.in_count; ++i)
        attrs[i] = {i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16u};
      vi.vertexBindingDescriptionCount = 1;
      vi.pVertexBindingDescriptions = &bind;
      vi.vertexAttributeDescriptionCount = vp->meta.in_count;
      vi.pVertexAttributeDescriptions = attrs;
    }
  } else {
    bind = {0, kVertexFloats * 4, VK_VERTEX_INPUT_RATE_VERTEX};
    for (uint32_t i = 0; i < 14; ++i) attrs[i] = {i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16u};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = 14;
    vi.pVertexAttributeDescriptions = attrs;
  }
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vps.viewportCount = 1;
  vps.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.depthClampEnable = depth_clamp_ ? VK_TRUE : VK_FALSE;  // the RSX clamps window z rather than clipping
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  if (vp) {
    if (!d.cull) rs.cullMode = VK_CULL_MODE_NONE;
    else if (d.cull_face == 0x404) rs.cullMode = VK_CULL_MODE_FRONT_BIT;
    else if (d.cull_face == 0x408) rs.cullMode = VK_CULL_MODE_FRONT_AND_BACK;
    else if (d.cull_face == 0x405) rs.cullMode = VK_CULL_MODE_BACK_BIT;
    else rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = d.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
  } else {
    rs.cullMode = VK_CULL_MODE_NONE;  // culling already happened on the CPU
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  }
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = with_depth && d.depth_test;
  ds.depthWriteEnable = with_depth && d.depth_write;
  ds.depthCompareOp = VkCompareOp(std::min<uint32_t>(d.depth_func - 0x200, 7));
  VkPipelineColorBlendAttachmentState cb{};
  cb.blendEnable = d.blend;
  cb.srcColorBlendFactor = blend_factor(d.sf_rgb);
  cb.dstColorBlendFactor = blend_factor(d.df_rgb);
  cb.colorBlendOp = blend_op(d.eq_rgb);
  cb.srcAlphaBlendFactor = blend_factor(d.sf_a);
  cb.dstAlphaBlendFactor = blend_factor(d.df_a);
  cb.alphaBlendOp = blend_op(d.eq_a);
  cb.colorWriteMask = ((d.color_mask & 0x00010000) ? VK_COLOR_COMPONENT_R_BIT : 0) |
                      ((d.color_mask & 0x00000100) ? VK_COLOR_COMPONENT_G_BIT : 0) |
                      ((d.color_mask & 0x00000001) ? VK_COLOR_COMPONENT_B_BIT : 0) |
                      ((d.color_mask & 0x01000000) ? VK_COLOR_COMPONENT_A_BIT : 0);
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1;
  cbs.pAttachments = &cb;
  const VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 3;
  dy.pDynamicStates = dyn;

  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 2;
  gp.pStages = stages;
  gp.pVertexInputState = &vi;
  gp.pInputAssemblyState = &ia;
  gp.pViewportState = &vps;
  gp.pRasterizationState = &rs;
  gp.pMultisampleState = &ms;
  gp.pDepthStencilState = &ds;
  gp.pColorBlendState = &cbs;
  gp.pDynamicState = &dy;
  gp.layout = pipeline_layout_;
  gp.renderPass = with_depth ? rp_depth_ : rp_color_;
  VkPipeline p = VK_NULL_HANDLE;
  if (vk_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &p) != VK_SUCCESS) {
    if (error) *error = "pipeline creation failed";
    return VK_NULL_HANDLE;
  }
  pipelines_[key] = p;
  return p;
}

// ============================================================================
// GPU surface reinterpretation (zero-readback depth/shadow/self-RT sampling)
// ============================================================================
VkPipeline VulkanRenderer::make_convert_pipeline(VkShaderModule fs, VkRenderPass rp) {
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs_;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";
  VkVertexInputBindingDescription bind{0, kVertexFloats * 4, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[14];
  for (uint32_t i = 0; i < 14; ++i) attrs[i] = {i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 14;
  vi.pVertexAttributeDescriptions = attrs;
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendAttachmentState cb{};
  cb.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo cbs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbs.attachmentCount = 1;
  cbs.pAttachments = &cb;
  const VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyn;
  VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 2;
  gp.pStages = stages;
  gp.pVertexInputState = &vi;
  gp.pInputAssemblyState = &ia;
  gp.pViewportState = &vp;
  gp.pRasterizationState = &rs;
  gp.pMultisampleState = &ms;
  gp.pDepthStencilState = &ds;
  gp.pColorBlendState = &cbs;
  gp.pDynamicState = &dy;
  gp.layout = pipeline_layout_;
  gp.renderPass = rp;
  VkPipeline p = VK_NULL_HANDLE;
  vk_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &p);
  return p;
}

bool VulkanRenderer::run_convert_pass(Surface& src, ConvertedImage& dst, VkPipeline pipe, VkRenderPass rp) {
  if (!pipe || !dst.fb) return false;
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = desc_pool_;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &set_layout_;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (vk_.vkAllocateDescriptorSets(device_, &dai, &set) != VK_SUCCESS) return false;
  ++sets_used_;
  VkDescriptorImageInfo infos[16];
  for (uint32_t u = 0; u < 16; ++u) infos[u] = {dummy_sampler_, dummy_.view, VK_IMAGE_LAYOUT_GENERAL};
  infos[0] = {dummy_sampler_, src.image.view, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorBufferInfo ubo{const_buf_.buf, 0, 16};
  VkWriteDescriptorSet writes[2]{};
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
               infos, nullptr, nullptr};
  writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 16, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
               nullptr, &ubo, nullptr};
  vk_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

  VkDeviceSize voff = 0;
  const VkDeviceSize vbytes = VkDeviceSize(3) * kVertexFloats * 4;
  if (!ring_alloc(vertex_, vbytes, voff)) return false;
  float* out = reinterpret_cast<float*>(vertex_.ptr + voff);
  std::memset(out, 0, size_t(vbytes));
  const float pos[3][2] = {{-1.0f, -1.0f}, {3.0f, -1.0f}, {-1.0f, 3.0f}};
  const float uv[3][2] = {{0.0f, 0.0f}, {2.0f, 0.0f}, {0.0f, 2.0f}};
  for (int i = 0; i < 3; ++i) {
    float* v = out + i * kVertexFloats;
    v[0] = pos[i][0]; v[1] = pos[i][1]; v[2] = 0.0f; v[3] = 1.0f;
    // TEX0 is varying index 3 -> attribute location 4 -> float offset 16.
    v[16] = uv[i][0]; v[17] = uv[i][1]; v[18] = 0.0f; v[19] = 1.0f;
  }

  begin();
  barrier_all();
  VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rbi.renderPass = rp;
  rbi.framebuffer = dst.fb;
  rbi.renderArea = {{0, 0}, {dst.image.w, dst.image.h}};
  vk_.vkCmdBeginRenderPass(cmd_, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  vk_.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkViewport vpt{0.0f, 0.0f, float(dst.image.w), float(dst.image.h), 0.0f, 1.0f};
  vk_.vkCmdSetViewport(cmd_, 0, 1, &vpt);
  VkRect2D sc{{0, 0}, {dst.image.w, dst.image.h}};
  vk_.vkCmdSetScissor(cmd_, 0, 1, &sc);
  vk_.vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set, 0, nullptr);
  vk_.vkCmdBindVertexBuffers(cmd_, 0, 1, &vertex_.buf, &voff);
  vk_.vkCmdDraw(cmd_, 3, 1, 0, 0);
  vk_.vkCmdEndRenderPass(cmd_);
  barrier_all();
  dst.write_seq = src.write_seq;
  return true;
}

bool VulkanRenderer::can_gpu_tex(const TexUnit& t, uint32_t target_color, uint32_t target_depth) const {
  if (t.is_3d()) return false;
  for (const auto& s : surfaces_) {
    if (!s->addr || s->needs_upload || s->addr != t.address || s->image.w != t.width || s->image.h != t.height)
      continue;
    if (!s->depth && s->argb && (t.format == 0x85 || t.format == 0x9E)) return true;  // direct or self-copy
    if (s->depth && t.is_depth_fmt() && s->addr != target_depth) return true;        // direct depth sample
    if (s->depth && t.format == 0x85 && s->depth_fmt == 2 && s->addr != target_depth && pipe_d24_argb_) return true;
    if (!s->depth && !s->argb && t.format == 0x95 && s->addr != target_color && pipe_abgr_y16x16_) return true;
  }
  return false;
}

VkImageView VulkanRenderer::resolve_gpu_tex(const TexUnit& t, Surface* active_color, Surface* active_depth,
                                            std::string* error) {
  if (t.is_3d()) return VK_NULL_HANDLE;
  for (auto& s : surfaces_) {
    if (!s->addr || s->needs_upload || s->addr != t.address || s->image.w != t.width || s->image.h != t.height)
      continue;
    // 1. ARGB color surface sampled as A8R8G8B8 / D8R8G8B8.
    if (!s->depth && s->argb && (t.format == 0x85 || t.format == 0x9E)) {
      if (s.get() != active_color) return (t.is_srgb() && s->image.srgb_view) ? s->image.srgb_view : s->image.view;
      // Self-feedback: copy active render target into a cached snapshot image on the GPU.
      auto& c = converted_[{s->addr, s->image.w, s->image.h, 0xFFu}];
      if (!c.image.img) {
        if (!make_image(c.image, s->image.w, s->image.h, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                        1, false, true)) {
          if (error) *error = "self-copy image allocation failed";
          return VK_NULL_HANDLE;
        }
        begin();
        transition_general(c.image.img, VK_IMAGE_ASPECT_COLOR_BIT);
      }
      if (c.write_seq != s->write_seq) {
        begin();
        barrier_all();
        VkImageCopy r{};
        r.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.extent = {s->image.w, s->image.h, 1};
        vk_.vkCmdCopyImage(cmd_, s->image.img, VK_IMAGE_LAYOUT_GENERAL, c.image.img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        barrier_all();
        c.write_seq = s->write_seq;
      }
      return (t.is_srgb() && c.image.srgb_view) ? c.image.srgb_view : c.image.view;
    }
    // 2. Depth surface sampled as depth format (0x8F..0x92): direct sample (swizzled R,R,R,1).
    if (s->depth && t.is_depth_fmt() && s.get() != active_depth) return s->image.view;
    // 3. Depth surface sampled as A8R8G8B8 (0x85, packed D24): GPU conversion pass.
    if (s->depth && t.format == 0x85 && s->depth_fmt == 2 && s.get() != active_depth && pipe_d24_argb_) {
      auto& c = converted_[{s->addr, s->image.w, s->image.h, 0x85u}];
      if (!c.image.img) {
        if (!make_image(c.image, s->image.w, s->image.h, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                        1, false, true)) {
          if (error) *error = "d24 convert image allocation failed";
          return VK_NULL_HANDLE;
        }
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = rp_color_;
        fci.attachmentCount = 1;
        fci.pAttachments = &c.image.view;
        fci.width = s->image.w;
        fci.height = s->image.h;
        fci.layers = 1;
        vk_.vkCreateFramebuffer(device_, &fci, nullptr, &c.fb);
        begin();
        transition_general(c.image.img, VK_IMAGE_ASPECT_COLOR_BIT);
      }
      if (c.write_seq != s->write_seq && !run_convert_pass(*s, c, pipe_d24_argb_, rp_color_)) return VK_NULL_HANDLE;
      return (t.is_srgb() && c.image.srgb_view) ? c.image.srgb_view : c.image.view;
    }
    // 4. ABGR color surface sampled as Y16_X16 (0x95, shadow atlas): GPU 16-bit conversion pass.
    if (!s->depth && !s->argb && t.format == 0x95 && s.get() != active_color && pipe_abgr_y16x16_) {
      auto& c = converted_[{s->addr, s->image.w, s->image.h, 0x95u}];
      if (!c.image.img) {
        if (!make_image(c.image, s->image.w, s->image.h, VK_FORMAT_R16G16_UNORM,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT)) {
          if (error) *error = "y16x16 convert image allocation failed";
          return VK_NULL_HANDLE;
        }
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = rp_r16g16_;
        fci.attachmentCount = 1;
        fci.pAttachments = &c.image.view;
        fci.width = s->image.w;
        fci.height = s->image.h;
        fci.layers = 1;
        vk_.vkCreateFramebuffer(device_, &fci, nullptr, &c.fb);
        begin();
        transition_general(c.image.img, VK_IMAGE_ASPECT_COLOR_BIT);
      }
      if (c.write_seq != s->write_seq && !run_convert_pass(*s, c, pipe_abgr_y16x16_, rp_r16g16_)) return VK_NULL_HANDLE;
      return c.image.view;
    }
  }
  return VK_NULL_HANDLE;
}

// ============================================================================
// draw / flush
// ============================================================================
bool VulkanRenderer::draw(const DrawEnv& env, const PreparedDraw& d, std::string* error) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  InRenderer in_renderer;
  if (!g_guard_base.load()) g_guard_base = env.mem;
  if (d.tris.empty()) return true;
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };

  const bool use_depth = d.depth_test || d.depth_write;
  const uint32_t target_depth = use_depth ? d.zbase : 0;

  // Texture units read guest memory: write back any GPU-rendered surface they overlap
  // before decoding them, unless we can bind or reinterpret the GPU surface directly.
  for (const TexUnit& t : d.tex) {
    if (!t.enabled || !t.base) continue;
    if (can_gpu_tex(t, d.color_base, target_depth)) continue;
    std::vector<Surface*> sources;
    if (composite_sources(t, d.color_base, sources)) continue;  // assembled on the GPU below
    const uint32_t size = t.byte_size();
    for (auto& s : surfaces_) {
      if (!s->addr) continue;
      if (s->dirty && t.address < s->addr + s->span() && s->addr < t.address + size) {
        static std::set<std::string> seen;
        char why[200];
        std::snprintf(why, sizeof why, "tex fmt 0x%02X %ux%u pitch %u @+0x%X vs %s surface %ux%u pitch %u argb %d%s%s",
                      t.format, t.width, t.height, t.pitch, unsigned(t.address - s->addr), s->depth ? "depth" : "color",
                      s->image.w, s->image.h, s->pitch, s->argb ? 1 : 0, s->needs_upload ? " needs_upload" : "",
                      s->addr == d.color_base ? " (is render target)" : "");
        if (seen.insert(why).second && seen.size() < 40) std::fprintf(stderr, "[gpu] flush for texture overlap: %s\n", why);
        flush_for(env.mem, s->depth, kFlushTextureOverlap, t.address, uint64_t(t.address) + size);
        break;
      }
    }
  }
  const VkDeviceSize vbytes_need = VkDeviceSize(d.tris.size()) * kVertexFloats * 4;
  const VkDeviceSize cbytes_need = VkDeviceSize(std::max<size_t>(1, d.fp_consts.size())) * 16;
  if (sets_used_ + 20 > kMaxSetsPerFlush || upload_.used > kUploadRing * 3 / 4 ||
      vertex_.used + vbytes_need + 4096 > kVertexRing || const_buf_.used + cbytes_need + 256 > kConstRing)
    flush_for(env.mem, false, kFlushRings, 0, 0);

  VkPipeline pipe = pipeline(d, use_depth, nullptr, error);
  if (!pipe) return false;

  const bool dummy_color = (d.color_mask == 0 && d.pitch < d.surface_w * 4);
  Surface* color = surface(env.mem, dummy_color ? 0u : d.color_base,
                           dummy_color ? d.surface_w * 4 : d.pitch,
                           d.surface_w, d.surface_h, false, 0, d.argb);
  if (!color) return fail("could not allocate a colour surface");
  if (dummy_color) color->needs_upload = false;
  Surface* depth = nullptr;
  if (use_depth) {
    const uint32_t bpp = d.depth_fmt == 1 ? 2 : 4;
    depth = surface(env.mem, d.zbase, d.zpitch, std::min(d.surface_w, d.zpitch / bpp), d.surface_h, true, d.depth_fmt, true);
    if (!depth) return fail("could not allocate a depth surface");
  }
  if (d.color_mask != 0 && color->needs_upload && !upload_surface(env.mem, *color)) { flush_for(env.mem, false, kFlushUpload, 0, 0); if (!upload_surface(env.mem, *color)) return fail("upload failed"); }
  if (depth && depth->needs_upload && !upload_surface(env.mem, *depth)) { flush_for(env.mem, false, kFlushUpload, 0, 0); if (!upload_surface(env.mem, *depth)) return fail("upload failed"); }

  // Fragment inline constants UBO (binding 16).
  VkDeviceSize coff = 0;
  if (!ring_alloc(const_buf_, cbytes_need, coff)) return fail("const ring full");
  if (d.fp_consts.empty()) std::memset(const_buf_.ptr + coff, 0, 16);
  else std::memcpy(const_buf_.ptr + coff, d.fp_consts.data(), d.fp_consts.size() * 16);

  // Descriptor set: every unit gets an image (the dummy when unused) + UBO at binding 16.
  VkDescriptorImageInfo infos[16];
  for (uint32_t u = 0; u < 16; ++u) {
    infos[u] = {dummy_sampler_, dummy_.view, VK_IMAGE_LAYOUT_GENERAL};
    const TexUnit& t = d.tex[u];
    if (!t.enabled || !t.base) continue;
    std::string terr;
    VkImageView v = resolve_gpu_tex(t, color, depth, &terr);
    if (!v && !terr.empty()) return fail(terr);
    if (!v) {
      std::vector<Surface*> sources;
      v = composite_sources(t, d.color_base, sources) ? composite(t, sources, &terr) : texture(t, &terr);
      if (!v) return fail(terr);
    }
    infos[u] = {sampler(t), v, VK_IMAGE_LAYOUT_GENERAL};
  }
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = desc_pool_;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &set_layout_;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (vk_.vkAllocateDescriptorSets(device_, &dai, &set) != VK_SUCCESS) return fail("descriptor pool exhausted");
  ++sets_used_;
  VkDescriptorBufferInfo ubo_info{const_buf_.buf, coff, cbytes_need};
  VkWriteDescriptorSet writes[2]{};
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
               infos, nullptr, nullptr};
  writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 16, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
               nullptr, &ubo_info, nullptr};
  vk_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

  // Vertices: window coordinates back to clip space for a viewport covering the surface.
  const uint32_t fw = depth ? std::min(color->image.w, depth->image.w) : color->image.w;
  const uint32_t fh = depth ? std::min(color->image.h, depth->image.h) : color->image.h;
  VkDeviceSize voff = 0;
  const VkDeviceSize vbytes = vbytes_need;
  if (!ring_alloc(vertex_, vbytes, voff)) return fail("vertex ring full");
  if (prof::enabled()) prof::record_call(10042, 0, vbytes / 1024, "gpu:vertex-KB");
  float* out = reinterpret_cast<float*>(vertex_.ptr + voff);
  for (const PreparedVertex& v : d.tris) {
    const float w = 1.0f / v.inv_w;
    *out++ = (2.0f * v.x / float(fw) - 1.0f) * w;
    *out++ = (2.0f * v.y / float(fh) - 1.0f) * w;
    *out++ = v.z * w;
    *out++ = w;
    for (int k = 0; k < kVaryings; ++k)
      for (int c = 0; c < 4; ++c) *out++ = v.vary[k][c];
  }

  begin();
  const VkRenderPass rp = depth ? rp_depth_ : rp_color_;
  const VkFramebuffer fb = framebuffer(color, depth);
  if (!pass_open_ || pass_fb_ != fb || pass_rp_ != rp || pass_w_ != fw || pass_h_ != fh) {
    barrier_all();  // also ends the previous pass
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp;
    rbi.framebuffer = fb;
    rbi.renderArea = {{0, 0}, {fw, fh}};
    vk_.vkCmdBeginRenderPass(cmd_, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pass_open_ = true;
    pass_fb_ = fb;
    pass_rp_ = rp;
    pass_w_ = fw;
    pass_h_ = fh;
  }
  const bool ts_this = ts_pool_ && prof::enabled() && 2 * ts_draws_.size() + 2 <= kTsQueries - 3;
  if (ts_this) vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, uint32_t(3 + 2 * ts_draws_.size()));
  vk_.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkViewport vpt{0.0f, 0.0f, float(fw), float(fh), 0.0f, 1.0f};
  vk_.vkCmdSetViewport(cmd_, 0, 1, &vpt);
  const int sx0 = std::max(d.min_x, 0), sy0 = std::max(d.min_y, 0);
  const int sx1 = std::min(d.max_x, int(fw)), sy1 = std::min(d.max_y, int(fh));
  VkRect2D sc{{sx0, sy0}, {uint32_t(std::max(0, sx1 - sx0)), uint32_t(std::max(0, sy1 - sy0))}};
  vk_.vkCmdSetScissor(cmd_, 0, 1, &sc);
  const float bc[4] = {((d.blend_color >> 16) & 0xFF) / 255.0f, ((d.blend_color >> 8) & 0xFF) / 255.0f,
                       (d.blend_color & 0xFF) / 255.0f, ((d.blend_color >> 24) & 0xFF) / 255.0f};
  vk_.vkCmdSetBlendConstants(cmd_, bc);
  vk_.vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set, 0, nullptr);
  vk_.vkCmdBindVertexBuffers(cmd_, 0, 1, &vertex_.buf, &voff);
  vk_.vkCmdDraw(cmd_, uint32_t(d.tris.size()), 1, 0, 0);
  ++draws_since_flush_;
  if (ts_this) {
    vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, uint32_t(4 + 2 * ts_draws_.size()));
    ts_draws_.push_back({uint32_t(d.fp_hash), uint32_t(d.tris.size() / 3), fw, fh});
  }

  if (d.color_mask != 0 && !dummy_color) mark_dirty(color);
  if (depth && d.depth_write) mark_dirty(depth);
  return true;
}

bool VulkanRenderer::draw_raw(const DrawEnv& env, const DrawCall& call, const PreparedDraw& d,
                              uint32_t& out_tris, std::string* error) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  InRenderer in_renderer;
  if (!g_guard_base.load()) g_guard_base = env.mem;
  out_tris = 0;
  auto fail = [&](const std::string& why) { if (error) *error = why; return false; };
  if (!env.vp) return fail("no vertex program bound");
  if (d.min_x >= d.max_x || d.min_y >= d.max_y) return true;

  if (call.primitive < 5 || call.primitive > 10) return fail("points and lines are not supported yet");
  size_t n_raw = 0;
  for (const auto& [first, count] : call.ranges) n_raw += count;
  if (n_raw < 3) return true;
  size_t max_idx_need = 0;
  switch (call.primitive) {
    case 5: max_idx_need = (n_raw / 3) * 3; break;
    case 6: case 7: case 10: max_idx_need = (n_raw - 2) * 3; break;
    case 8: max_idx_need = (n_raw / 4) * 6; break;
    case 9: max_idx_need = n_raw >= 4 ? ((n_raw - 2) / 2) * 6 : 0; break;
    default: break;
  }
  if (max_idx_need == 0) return true;

  const CachedVp* vp = get_vp(*env.vp, error);
  if (!vp || !vp->ok) return false;

  const bool use_depth = d.depth_test || d.depth_write;
  const uint32_t target_depth = use_depth ? d.zbase : 0;

  for (const TexUnit& t : d.tex) {
    if (!t.enabled || !t.base) continue;
    if (can_gpu_tex(t, d.color_base, target_depth)) continue;
    std::vector<Surface*> sources;
    if (composite_sources(t, d.color_base, sources)) continue;
    const uint32_t size = t.byte_size();
    for (auto& s : surfaces_) {
      if (!s->addr) continue;
      if (s->dirty && t.address < s->addr + s->span() && s->addr < t.address + size) {
        flush_for(env.mem, s->depth, kFlushTextureOverlap, t.address, uint64_t(t.address) + size);
        break;
      }
    }
  }

  const VkDeviceSize vtx_stride = VkDeviceSize(vp->meta.in_count) * 16;
  const VkDeviceSize vbytes_max = VkDeviceSize(n_raw) * vtx_stride;
  const VkDeviceSize ibytes_max = VkDeviceSize(max_idx_need) * 4;
  const VkDeviceSize cbytes_fp = VkDeviceSize(std::max<size_t>(1, d.fp_consts.size())) * 16;
  if (sets_used_ + 20 > kMaxSetsPerFlush || upload_.used > kUploadRing * 3 / 4 ||
      vertex_.used + vbytes_max + 4096 > kVertexRing || index_.used + ibytes_max + 4096 > kIndexRing ||
      const_buf_.used + cbytes_fp + kVpUboSize + 1024 > kConstRing)
    flush_for(env.mem, false, kFlushRings, 0, 0);

  VkPipeline pipe = pipeline(d, use_depth, vp, error);
  if (!pipe) return false;

  const bool dummy_color = (d.color_mask == 0 && d.pitch < d.surface_w * 4);
  Surface* color = surface(env.mem, dummy_color ? 0u : d.color_base,
                           dummy_color ? d.surface_w * 4 : d.pitch,
                           d.surface_w, d.surface_h, false, 0, d.argb);
  if (!color) return fail("could not allocate a colour surface");
  if (dummy_color) color->needs_upload = false;
  Surface* depth = nullptr;
  if (use_depth) {
    const uint32_t bpp = d.depth_fmt == 1 ? 2 : 4;
    depth = surface(env.mem, d.zbase, d.zpitch, std::min(d.surface_w, d.zpitch / bpp), d.surface_h, true, d.depth_fmt, true);
    if (!depth) return fail("could not allocate a depth surface");
  }
  if (d.color_mask != 0 && color->needs_upload && !upload_surface(env.mem, *color)) {
    flush_for(env.mem, false, kFlushUpload, 0, 0);
    if (!upload_surface(env.mem, *color)) return fail("upload failed");
  }
  if (depth && depth->needs_upload && !upload_surface(env.mem, *depth)) {
    flush_for(env.mem, false, kFlushUpload, 0, 0);
    if (!upload_surface(env.mem, *depth)) return fail("upload failed");
  }

  VkDescriptorImageInfo infos[16];
  for (uint32_t u = 0; u < 16; ++u) {
    infos[u] = {dummy_sampler_, dummy_.view, VK_IMAGE_LAYOUT_GENERAL};
    const TexUnit& t = d.tex[u];
    if (!t.enabled || !t.base) continue;
    std::string terr;
    VkImageView v = resolve_gpu_tex(t, color, depth, &terr);
    if (!v && !terr.empty()) return fail(terr);
    if (!v) {
      std::vector<Surface*> sources;
      v = composite_sources(t, d.color_base, sources) ? composite(t, sources, &terr) : texture(t, &terr);
      if (!v) return fail(terr);
    }
    infos[u] = {sampler(t), v, VK_IMAGE_LAYOUT_GENERAL};
  }

  // Pack unique vertices and expanded triangle-list indices straight into the GPU rings.
  const VkDeviceSize voff = (vertex_.used + 255) & ~VkDeviceSize(255);
  const VkDeviceSize ioff = (index_.used + 255) & ~VkDeviceSize(255);
  const uint32_t avail_vtx = vtx_stride
      ? uint32_t(std::min<VkDeviceSize>(n_raw, vertex_.size > voff ? (vertex_.size - voff) / vtx_stride : 0))
      : uint32_t(n_raw);
  const uint32_t avail_idx =
      uint32_t(std::min<VkDeviceSize>(max_idx_need, index_.size > ioff ? (index_.size - ioff) / 4 : 0));
  if (avail_idx < max_idx_need || (vtx_stride && avail_vtx == 0)) return fail("vertex/index ring full");
  uint32_t n_vtx = 0, n_idx = 0;
  if (!pack_indexed_geometry(env, call, vp->meta.in_mask, vp->meta.in_count,
                             vtx_stride ? reinterpret_cast<float*>(vertex_.ptr + voff) : nullptr, avail_vtx, n_vtx,
                             reinterpret_cast<uint32_t*>(index_.ptr + ioff), avail_idx, n_idx, error))
    return false;
  if (n_idx == 0) return true;
  if (vtx_stride) vertex_.used = voff + VkDeviceSize(n_vtx) * vtx_stride;
  index_.used = ioff + VkDeviceSize(n_idx) * 4;
  out_tris = n_idx / 3;
  if (prof::enabled())
    prof::record_call(10042, 0, (VkDeviceSize(n_vtx) * vtx_stride + VkDeviceSize(n_idx) * 4) / 1024, "gpu:vertex-KB");

  // Fragment inline constants UBO (binding 16).
  VkDeviceSize coff = 0;
  if (!ring_alloc(const_buf_, cbytes_fp, coff)) return fail("const ring full");
  if (d.fp_consts.empty()) std::memset(const_buf_.ptr + coff, 0, 16);
  else std::memcpy(const_buf_.ptr + coff, d.fp_consts.data(), d.fp_consts.size() * 16);

  // Vertex viewport & 512 constants UBO (binding 17).
  const uint32_t fw = depth ? std::min(color->image.w, depth->image.w) : color->image.w;
  const uint32_t fh = depth ? std::min(color->image.h, depth->image.h) : color->image.h;
  const float vp_s[4] = {
      2.0f * float(d.aa_x) * d.vp_scale[0] / float(fw),
      2.0f * float(d.aa_y) * d.vp_scale[1] / float(fh),
      d.vp_scale[2],
      0.0f,
  };
  const float vp_o[4] = {
      2.0f * float(d.aa_x) * d.vp_offset[0] / float(fw) - 1.0f,
      2.0f * float(d.aa_y) * d.vp_offset[1] / float(fh) - 1.0f,
      d.vp_offset[2],
      0.0f,
  };
  VkDeviceSize vpoff = 0;
  if (env.vp->const_ver > 1 && last_vp_ubo_valid_ && last_vp_ubo_ptr_ == env.vp &&
      last_vp_ubo_ver_ == env.vp->const_ver && std::memcmp(last_vp_s_, vp_s, 16) == 0 &&
      std::memcmp(last_vp_o_, vp_o, 16) == 0) {
    vpoff = last_vp_ubo_off_;
  } else {
    if (!ring_alloc(const_buf_, kVpUboSize, vpoff)) return fail("const ring full");
    uint8_t* dst = const_buf_.ptr + vpoff;
    std::memcpy(dst, vp_s, 16);
    std::memcpy(dst + 16, vp_o, 16);
    const size_t n_c = std::min<size_t>(512, env.vp->constants.size());
    std::memcpy(dst + 32, env.vp->constants.data(), n_c * 16);
    if (n_c < 512) std::memset(dst + 32 + n_c * 16, 0, (512 - n_c) * 16);
    if (env.vp->const_ver > 1) {
      last_vp_ubo_ptr_ = env.vp;
      last_vp_ubo_ver_ = env.vp->const_ver;
      std::memcpy(last_vp_s_, vp_s, 16);
      std::memcpy(last_vp_o_, vp_o, 16);
      last_vp_ubo_off_ = vpoff;
      last_vp_ubo_valid_ = true;
    }
  }

  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = desc_pool_;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &set_layout_;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (vk_.vkAllocateDescriptorSets(device_, &dai, &set) != VK_SUCCESS) return fail("descriptor pool exhausted");
  ++sets_used_;
  VkDescriptorBufferInfo fp_ubo{const_buf_.buf, coff, cbytes_fp};
  VkDescriptorBufferInfo vp_ubo{const_buf_.buf, vpoff, kVpUboSize};
  VkWriteDescriptorSet writes[3]{};
  writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
               infos, nullptr, nullptr};
  writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 16, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
               nullptr, &fp_ubo, nullptr};
  writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 17, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
               nullptr, &vp_ubo, nullptr};
  vk_.vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

  begin();
  const VkRenderPass rp = depth ? rp_depth_ : rp_color_;
  const VkFramebuffer fb = framebuffer(color, depth);
  if (!pass_open_ || pass_fb_ != fb || pass_rp_ != rp || pass_w_ != fw || pass_h_ != fh) {
    barrier_all();
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = rp;
    rbi.framebuffer = fb;
    rbi.renderArea = {{0, 0}, {fw, fh}};
    vk_.vkCmdBeginRenderPass(cmd_, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    pass_open_ = true;
    pass_fb_ = fb;
    pass_rp_ = rp;
    pass_w_ = fw;
    pass_h_ = fh;
  }
  const bool ts_this = ts_pool_ && prof::enabled() && 2 * ts_draws_.size() + 2 <= kTsQueries - 3;
  if (ts_this) vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, uint32_t(3 + 2 * ts_draws_.size()));
  vk_.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkViewport vpt{0.0f, 0.0f, float(fw), float(fh), 0.0f, 1.0f};
  vk_.vkCmdSetViewport(cmd_, 0, 1, &vpt);
  const int sx0 = std::max(d.min_x, 0), sy0 = std::max(d.min_y, 0);
  const int sx1 = std::min(d.max_x, int(fw)), sy1 = std::min(d.max_y, int(fh));
  VkRect2D sc{{sx0, sy0}, {uint32_t(std::max(0, sx1 - sx0)), uint32_t(std::max(0, sy1 - sy0))}};
  vk_.vkCmdSetScissor(cmd_, 0, 1, &sc);
  const float bc[4] = {((d.blend_color >> 16) & 0xFF) / 255.0f, ((d.blend_color >> 8) & 0xFF) / 255.0f,
                       (d.blend_color & 0xFF) / 255.0f, ((d.blend_color >> 24) & 0xFF) / 255.0f};
  vk_.vkCmdSetBlendConstants(cmd_, bc);
  vk_.vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set, 0, nullptr);
  if (vp->meta.in_count > 0) vk_.vkCmdBindVertexBuffers(cmd_, 0, 1, &vertex_.buf, &voff);
  vk_.vkCmdBindIndexBuffer(cmd_, index_.buf, ioff, VK_INDEX_TYPE_UINT32);
  vk_.vkCmdDrawIndexed(cmd_, n_idx, 1, 0, 0, 0);
  ++draws_since_flush_;
  if (ts_this) {
    vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, uint32_t(4 + 2 * ts_draws_.size()));
    ts_draws_.push_back({uint32_t(d.fp_hash), out_tris, fw, fh});
  }

  if (d.color_mask != 0 && !dummy_color) mark_dirty(color);
  if (depth && d.depth_write) mark_dirty(depth);
  return true;
}

bool VulkanRenderer::clear(const DrawEnv& env, uint32_t mask) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  InRenderer in_renderer;
  if (!g_guard_base.load()) g_guard_base = env.mem;
  const uint32_t color_bits = mask & 0xF0;
  const uint32_t depth_bits = mask & 0x03;
  if (!color_bits && !depth_bits) return true;
  if (color_bits && color_bits != 0xF0) return false;  // partial channel mask falls back to CPU

  const uint32_t* r = env.regs;
  const uint32_t surf = r[m::SURFACE_FORMAT];
  const uint32_t color_fmt = surf & 0x1F, surf_type = (surf >> 8) & 0xF, depth_fmt = (surf >> 5) & 7;
  if (surf_type == 2) return false;
  const bool argb = color_fmt == 8 || color_fmt == 4 || color_fmt == 5;
  const bool abgr = color_fmt == 16 || color_fmt == 14 || color_fmt == 15;
  if (color_bits && !argb && !abgr) return false;

  uint32_t aa_x = 1, aa_y = 1;
  surface_aa_scale(surf, aa_x, aa_y);
  const uint32_t clip_h = r[m::SURFACE_CLIP_H], clip_v = r[m::SURFACE_CLIP_V];
  const uint32_t x0 = (clip_h & 0xFFFF) * aa_x, w = (clip_h >> 16) * aa_x;
  const uint32_t y0 = (clip_v & 0xFFFF) * aa_y, h = (clip_v >> 16) * aa_y;
  if (!w || !h) return true;
  if (x0 != 0 || y0 != 0) return false;  // sub-rect clears fall back to CPU
  const uint32_t surf_w_full = x0 + w, surf_h = y0 + h;

  Surface* color = nullptr;
  if (color_bits) {
    const uint32_t pitch = r[m::SURFACE_PITCH_A];
    const uint32_t cbase = env.to_address(r[m::SURFACE_COLOR_AOFFSET], r[m::CONTEXT_DMA_COLOR_A] == 0xFEED0001 ? 1 : 0);
    if (!cbase || pitch < 4) return false;
    const uint32_t sw = std::min(surf_w_full, pitch / 4);
    color = surface(env.mem, cbase, pitch, sw, surf_h, false, 0, argb);
    if (!color) return false;
    color->needs_upload = false;
  }

  Surface* depth = nullptr;
  if (depth_bits & 1) {
    const uint32_t zpitch = r[m::SURFACE_PITCH_Z], cpitch = r[m::SURFACE_PITCH_A];
    const uint32_t zbase = env.to_address(r[m::SURFACE_ZETA_OFFSET], r[m::CONTEXT_DMA_ZETA] == 0xFEED0001 ? 1 : 0);
    const uint32_t bpp = depth_fmt == 1 ? 2 : 4;
    if (zbase && zpitch >= bpp) {
      const uint32_t base_w = (color_bits && cpitch) ? std::min(surf_w_full, cpitch / 4) : surf_w_full;
      const uint32_t zw = std::min(base_w, zpitch / bpp);
      depth = surface(env.mem, zbase, zpitch, zw, surf_h, true, depth_fmt, true);
      if (!depth) return false;
      depth->needs_upload = false;
    }
  }

  begin();
  barrier_all();
  if (color) {
    const uint32_t cv = r[m::COLOR_CLEAR_VALUE];
    VkClearColorValue ccv{};
    if (argb) {
      ccv.float32[0] = ((cv >> 16) & 0xFF) / 255.0f;
      ccv.float32[1] = ((cv >> 8) & 0xFF) / 255.0f;
      ccv.float32[2] = (cv & 0xFF) / 255.0f;
      ccv.float32[3] = (cv >> 24) / 255.0f;
    } else {
      ccv.float32[0] = (cv & 0xFF) / 255.0f;
      ccv.float32[1] = ((cv >> 8) & 0xFF) / 255.0f;
      ccv.float32[2] = ((cv >> 16) & 0xFF) / 255.0f;
      ccv.float32[3] = (cv >> 24) / 255.0f;
    }
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk_.vkCmdClearColorImage(cmd_, color->image.img, VK_IMAGE_LAYOUT_GENERAL, &ccv, 1, &range);
    mark_dirty(color);
  }
  if (depth) {
    const uint32_t zv = r[m::ZSTENCIL_CLEAR_VALUE];
    VkClearDepthStencilValue dsv{};
    dsv.depth = depth_fmt == 1 ? uint16_t(zv) / 65535.0f : (zv >> 8) / 16777215.0f;
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vk_.vkCmdClearDepthStencilImage(cmd_, depth->image.img, VK_IMAGE_LAYOUT_GENERAL, &dsv, 1, &range);
    mark_dirty(depth);
  }
  barrier_all();
  return true;
}

void VulkanRenderer::flush_for(uint8_t* mem, bool include_depth, FlushReason why, uint64_t lo, uint64_t hi) {
  const auto t0 = std::chrono::steady_clock::now();
  static const bool full_sync = std::getenv("PS3RECOMP_FULL_SYNC") != nullptr;
  if (full_sync) { lo = 0; hi = ~0ull; }
  flush_impl(mem, include_depth, lo, hi);
  if (prof::enabled())
    prof::record_call(10020 + why, 0,
                      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
                      "gpu:flush");
}

void VulkanRenderer::record_readbacks(uint64_t lo, uint64_t hi, bool include_depth) {
  // Copy every requested GPU-modified surface into the readback ring. Also include
  // small dirty main-RAM colour targets (e.g. 32x16 auto-exposure/luminance buffers)
  // during frame flips so CPU reads never trigger synchronous guard-page stalls.
  pending_readback_.clear();
  for (auto& s : surfaces_) {
    if (!s->dirty || (s->depth && !include_depth)) continue;
    const bool in_range = s->addr < hi && uint64_t(s->addr) + s->span() > lo;
    const bool small_main_rt = !include_depth && !s->depth && s->addr && s->addr < 0xC0000000u && s->span() <= 65536u;
    if (!in_range && !small_main_rt) continue;
    VkDeviceSize off = 0;
    if (!ring_alloc(readback_, VkDeviceSize(s->image.w) * s->image.h * 4, off)) continue;
    VkBufferImageCopy c{};
    c.bufferOffset = off;
    c.imageSubresource = {VkImageAspectFlags(s->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
    c.imageExtent = {s->image.w, s->image.h, 1};
    vk_.vkCmdCopyImageToBuffer(cmd_, s->image.img, VK_IMAGE_LAYOUT_GENERAL, readback_.buf, 1, &c);
    pending_readback_.push_back({s.get(), off});
  }
}

// Writes readbacks into guest memory in the guest's formats.
void VulkanRenderer::write_back(uint8_t* mem, const Buffer& readback,
                                const std::vector<std::pair<Surface*, VkDeviceSize>>& pending) {
  for (auto& [s, off] : pending) {
    if (s->guarded) guard_surface(*s, false);
    const uint8_t* src = readback.ptr + off;
    const uint32_t w = s->image.w, h = s->image.h;
    if (s->depth) {
      for (uint32_t y = 0; y < h; ++y) {
        uint8_t* row = mem + s->addr + size_t(y) * s->pitch;
        for (uint32_t x = 0; x < w; ++x) {
          const uint8_t* p = src + (size_t(y) * w + x) * 4;
          float z;
          std::memcpy(&z, p, 4);
          z = std::clamp(z, 0.0f, 1.0f);
          if (s->depth_fmt == 1) put_be16(row + 2 * x, uint16_t(z * 65535.0f + 0.5f));
          else put_be32(row + 4 * x, (uint32_t(z * 16777215.0f + 0.5f) << 8) | (be32(row + 4 * x) & 0xFF));
        }
      }
    } else {
      const __m128i shuf = s->argb
          ? _mm_setr_epi8(3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14)
          : _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
      for (uint32_t y = 0; y < h; ++y) {
        uint8_t* row = mem + s->addr + size_t(y) * s->pitch;
        const uint8_t* src_row = src + size_t(y) * w * 4;
        uint32_t x = 0;
        for (; x + 4 <= w; x += 4) {
          const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_row + 4 * x));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(row + 4 * x), _mm_shuffle_epi8(v, shuf));
        }
        for (; x < w; ++x) {
          const uint8_t* p = src_row + 4 * x;
          if (s->argb) {
            put_be32(row + 4 * x, (uint32_t(p[3]) << 24) | (uint32_t(p[0]) << 16) | (uint32_t(p[1]) << 8) | p[2]);
          } else {
            put_be32(row + 4 * x, (uint32_t(p[3]) << 24) | (uint32_t(p[2]) << 16) | (uint32_t(p[1]) << 8) | p[0]);
          }
        }
      }
    }
  }
}

void VulkanRenderer::evict_textures() {
  ++flush_seq_;
  for (auto it = textures_.begin(); it != textures_.end();) {
    if (it->second.dynamic_rt && flush_seq_ - it->second.last_flush > 3) {
      free_image(it->second.image);
      it = textures_.erase(it);
    } else {
      ++it;
    }
  }
  if (textures_.size() <= 4096) return;
  // Evict textures not used in recent flushes instead of wiping the entire cache.
  for (auto it = textures_.begin(); it != textures_.end();) {
    if (flush_seq_ - it->second.last_flush > 60) {
      free_image(it->second.image);
      it = textures_.erase(it);
    } else {
      ++it;
    }
  }
  if (textures_.size() > 4096) {
    vk_.vkDeviceWaitIdle(device_);
    for (auto& [k, t] : textures_) free_image(t.image);
    textures_.clear();
  }
}

void VulkanRenderer::swap_frame() {
  std::swap(cmd_, other_.cmd);
  std::swap(fence_, other_.fence);
  std::swap(upload_, other_.upload);
  std::swap(vertex_, other_.vertex);
  std::swap(index_, other_.index);
  std::swap(const_buf_, other_.const_buf);
  std::swap(readback_, other_.readback);
  std::swap(desc_pool_, other_.desc_pool);
  std::swap(sets_used_, other_.sets_used);
  std::swap(ts_pool_, other_.ts_pool);
  std::swap(ts_draws_, other_.ts_draws);
  std::swap(pending_readback_, other_.pending);
  std::swap(draws_since_flush_, other_.draws);
  last_vp_ubo_valid_ = false;
}

// Waits for a submitted frame, writes its readbacks to guest memory, runs its
// completion callback and makes its resources reusable.
void VulkanRenderer::complete(FrameState& f) {
  if (!f.in_flight) return;
  f.in_flight = false;
  vk_.vkWaitForFences(device_, 1, &f.fence, VK_TRUE, UINT64_MAX);
  if (prof::enabled()) {
    prof::record_call(10035, 0,
                      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - f.t_submit).count()),
                      "gpu:async-frame-latency");
    uint64_t ts[3] = {};
    if (f.ts_pool && vk_.vkGetQueryPoolResults(device_, f.ts_pool, 0, 3, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      prof::record_call(10033, 0, uint64_t(double(ts[1] - ts[0]) * ts_period_ns_ / 1000.0), "gpu:exec-frame-work");
      prof::record_call(10034, uint32_t(f.pending.size()), uint64_t(double(ts[2] - ts[1]) * ts_period_ns_ / 1000.0),
                        "gpu:exec-readback-copies");
    }
  }
  if (f.mem) write_back(f.mem, f.readback, f.pending);
  f.pending.clear();
  f.ts_draws.clear();
  f.upload.used = f.vertex.used = f.index.used = f.const_buf.used = f.readback.used = 0;
  vk_.vkResetDescriptorPool(device_, f.desc_pool, 0);
  f.sets_used = 0;
  f.draws = 0;
  auto done = std::move(f.done);
  f.done = nullptr;
  if (done) done();
}

void VulkanRenderer::flip_async(uint8_t* mem, uint32_t addr, uint32_t size, std::function<void()> done) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  if (!recording_) {
    // Nothing new on the GPU: just make sure earlier frames are finished.
    drain();
    for (auto& s : surfaces_)
      if (s->dirty && !s->depth && s->addr < uint64_t(addr) + size && s->addr + s->span() > addr) {
        flush_impl(mem, false, addr, uint64_t(addr) + size);
        break;
      }
    if (done) done();
    return;
  }
  end_pass();
  if (ts_pool_ && prof::enabled()) {
    barrier_all();
    vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, 1);
  }
  static const bool full_sync = std::getenv("PS3RECOMP_FULL_SYNC") != nullptr;
  record_readbacks(full_sync ? 0 : addr, full_sync ? ~0ull : uint64_t(addr) + size, false);
  barrier_all();
  if (ts_pool_ && prof::enabled()) vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, 2);
  vk_.vkEndCommandBuffer(cmd_);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vk_.vkResetFences(device_, 1, &fence_);
  vk_.vkQueueSubmit(queue_, 1, &si, fence_);
  recording_ = false;
  // The data reaches guest memory when the frame completes; later draws to these
  // surfaces mark them dirty again.
  for (auto& [s, off] : pending_readback_) s->dirty = false;

  drain();       // the older frame must finish before its slot is reused
  swap_frame();  // other_ = the frame just submitted; current = free slot
  other_.in_flight = true;
  other_.mem = mem;
  other_.done = std::move(done);
  other_.t_submit = std::chrono::steady_clock::now();
  evict_textures();
}

void VulkanRenderer::mark_dirty(Surface* s) {
  s->dirty = true;
  ++s->write_seq;
  if (!s->guarded && guards_enabled() && g_guard_base.load()) guard_surface(*s, true);
}

// Protects (on) or releases (off) the guest pages of `s`. Releasing re-protects
// pages still covered by another guarded surface.
void VulkanRenderer::guard_surface(Surface& s, bool on) {
#if defined(_WIN32)
  uint8_t* base = g_guard_base.load();
  if (!base || !s.span()) return;
  const uint32_t first = s.addr >> 12, last = uint32_t((uint64_t(s.addr) + s.span() - 1) >> 12);
  DWORD old = 0;
  if (on) {
    set_page_bits(first, last, true);
    VirtualProtect(base + (uint64_t(first) << 12), (uint64_t(last - first) + 1) << 12, PAGE_NOACCESS, &old);
    s.guarded = true;
    return;
  }
  s.guarded = false;
  VirtualProtect(base + (uint64_t(first) << 12), (uint64_t(last - first) + 1) << 12, PAGE_READWRITE, &old);
  set_page_bits(first, last, false);
  for (auto& o : surfaces_) {
    if (o.get() == &s || !o->guarded) continue;
    const uint32_t of = o->addr >> 12, ol = uint32_t((uint64_t(o->addr) + o->span() - 1) >> 12);
    if (of <= last && first <= ol) guard_surface(*o, true);
  }
#else
  s.guarded = on;
#endif
}

void VulkanRenderer::on_guard_fault(uint32_t guest_addr, bool write) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  const uint32_t page = guest_addr >> 12;
  if (!page_guarded(page)) return;  // released while we waited for the lock
  const uint64_t lo = uint64_t(page) << 12, hi = lo + 4096;
  static std::atomic<int> faults{0}, stale{0};
  if (t_in_renderer) {
    // The renderer itself is reading these pages (e.g. a texture or vertex fetch
    // overlapping a render target): no readback from inside a draw.
    if (stale++ < 8) logf("[gpu] lazy readback: renderer touched GPU-dirty page 0x%08X; using guest memory as is", guest_addr);
  } else {
    if (faults++ < 16)
      logf("[gpu] lazy readback: %s of 0x%08X", write ? "write" : "read", guest_addr);
    flush_impl(g_guard_base.load(), true, lo, hi);
  }
  // Whatever is still guarded on this page (not dirty, or touched from inside the
  // renderer) is released so the access can proceed.
  for (auto& s : surfaces_) {
    if (!s->guarded) continue;
    const uint32_t first = s->addr >> 12, last = uint32_t((uint64_t(s->addr) + s->span() - 1) >> 12);
    if (first <= page && page <= last) guard_surface(*s, false);
  }
  if (write) {
    // The CPU is changing a render target: the GPU copy must be reloaded.
    for (auto& s : surfaces_)
      if (s->addr < hi && uint64_t(s->addr) + s->span() > lo) s->needs_upload = true;
  }
#if defined(_WIN32)
  if (page_guarded(page)) {  // a page shared with no surface any more
    DWORD old = 0;
    VirtualProtect(g_guard_base.load() + lo, 4096, PAGE_READWRITE, &old);
    set_page_bits(page, page, false);
  }
#endif
}

#if defined(_WIN32)
LONG CALLBACK guard_fault_handler(EXCEPTION_POINTERS* ep) {
  const EXCEPTION_RECORD* er = ep->ExceptionRecord;
  if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
  uint8_t* base = g_guard_base.load();
  VulkanRenderer* r = g_guard_renderer.load();
  const uintptr_t a = uintptr_t(er->ExceptionInformation[1]);
  if (!base || !r || a < uintptr_t(base) || a - uintptr_t(base) >= (uint64_t(1) << 32)) return EXCEPTION_CONTINUE_SEARCH;
  const uint32_t ga = uint32_t(a - uintptr_t(base));
  if (!page_guarded(ga >> 12)) {
    // Race with guard_surface(s, false): another thread may have just restored
    // PAGE_READWRITE and cleared the guard bit while we were entering the VEH.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(a), &mbi, sizeof mbi) && mbi.State == MEM_COMMIT) {
      const DWORD prot = mbi.Protect & 0xFFu;
      const bool is_write = er->ExceptionInformation[0] == 1;
      const bool ok = is_write ? (prot == PAGE_READWRITE || prot == PAGE_EXECUTE_READWRITE || prot == PAGE_WRITECOPY)
                               : (prot == PAGE_READWRITE || prot == PAGE_READONLY || prot == PAGE_EXECUTE_READ ||
                                  prot == PAGE_EXECUTE_READWRITE || prot == PAGE_WRITECOPY);
      if (ok) return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
  }
  r->on_guard_fault(ga, er->ExceptionInformation[0] == 1);
  return EXCEPTION_CONTINUE_EXECUTION;
}
#endif

void VulkanRenderer::flush_impl(uint8_t* mem, bool include_depth, uint64_t lo, uint64_t hi) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  drain();  // finish the previous (older) frame first: its write-backs come before ours
  bool any_readback = false;
  for (auto& s : surfaces_) {
    if (s->dirty && (include_depth || !s->depth) && s->addr < hi && uint64_t(s->addr) + s->span() > lo) {
      any_readback = true;
      break;
    }
  }
  if (!any_readback && (!recording_ || lo > 0 || hi < ~0ull)) return;
  if (!recording_) begin();
  end_pass();

  if (ts_pool_ && prof::enabled()) {
    barrier_all();
    vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, 1);
  }
  record_readbacks(lo, hi, include_depth);
  barrier_all();
  const bool timed = ts_pool_ && prof::enabled();
  if (timed) vk_.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, 2);
  vk_.vkEndCommandBuffer(cmd_);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  vk_.vkResetFences(device_, 1, &fence_);
  const auto t_submit = std::chrono::steady_clock::now();
  vk_.vkQueueSubmit(queue_, 1, &si, fence_);
  vk_.vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  recording_ = false;
  const auto t_done = std::chrono::steady_clock::now();
  if (prof::enabled()) {
    prof::record_call(10030, uint32_t(pending_readback_.size()),
                      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(t_done - t_submit).count()), "gpu:submit+wait");
    uint64_t ts[3] = {};
    if (timed && vk_.vkGetQueryPoolResults(device_, ts_pool_, 0, 3, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      prof::record_call(10033, 0, uint64_t(double(ts[1] - ts[0]) * ts_period_ns_ / 1000.0), "gpu:exec-frame-work");
      prof::record_call(10034, uint32_t(pending_readback_.size()), uint64_t(double(ts[2] - ts[1]) * ts_period_ns_ / 1000.0),
                        "gpu:exec-readback-copies");
    }
  }
  ts_draws_.clear();
  draws_since_flush_ = 0;

  write_back(mem, readback_, pending_readback_);
  for (auto& [s, off] : pending_readback_) s->dirty = false;
  if (prof::enabled())
    prof::record_call(10031, 0,
                      uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_done).count()),
                      "gpu:readback-convert");
  pending_readback_.clear();
  upload_.used = vertex_.used = index_.used = const_buf_.used = readback_.used = 0;
  last_vp_ubo_valid_ = false;
  vk_.vkResetDescriptorPool(device_, desc_pool_, 0);
  sets_used_ = 0;
  evict_textures();
}

}  // namespace

std::unique_ptr<GpuRenderer> GpuRenderer::create(std::function<void(std::string_view)> log) {
  auto r = std::make_unique<VulkanRenderer>(log);
  if (!r->init()) {
    if (log) log("rsx: Vulkan renderer unavailable, using the CPU renderer");
    return nullptr;
  }
  return r;
}

}  // namespace rsx
