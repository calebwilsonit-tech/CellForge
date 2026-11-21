// Vulkan presenter: a Win32 window plus a swapchain. Frames arrive as BGRA pixels,
// are uploaded into an image and blitted (scaled) onto the swapchain image.
// The Vulkan loader (vulkan-1.dll, installed by the GPU driver) is loaded at runtime.

#include "presenter.h"

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace gfx {
std::atomic<int> g_capture_requests{0};
std::atomic<int> g_debug_f2{0}, g_debug_f3{0}, g_debug_f5{0}, g_debug_f6{0}, g_debug_f7{0}, g_debug_f8{0}, g_debug_f9{0}, g_debug_f11{0}, g_debug_f12{0};
namespace {

#define VK_GLOBAL_FUNCS(X) X(vkCreateInstance)
#define VK_INSTANCE_FUNCS(X)                                                                          \
  X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                 \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) \
  X(vkGetDeviceProcAddr) X(vkCreateWin32SurfaceKHR) X(vkDestroySurfaceKHR)                             \
  X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)                 \
  X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR)
#define VK_DEVICE_FUNCS(X)                                                                             \
  X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR)              \
  X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) X(vkQueueSubmit)            \
  X(vkDeviceWaitIdle) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers)       \
  X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkCmdPipelineBarrier)        \
  X(vkCmdCopyBufferToImage) X(vkCmdBlitImage) X(vkCmdClearColorImage) X(vkCreateBuffer)                \
  X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory)              \
  X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkCreateImage) X(vkDestroyImage)             \
  X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateSemaphore) X(vkDestroySemaphore)      \
  X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)

struct Vk {
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define DECL(f) PFN_##f f = nullptr;
  VK_GLOBAL_FUNCS(DECL) VK_INSTANCE_FUNCS(DECL) VK_DEVICE_FUNCS(DECL)
#undef DECL
};

constexpr wchar_t kClass[] = L"cellforge_window";

HICON load_png_icon(const std::string& path) {
  if (path.empty()) return nullptr;
  struct GdiInit {
    ULONG_PTR token = 0;
    GdiInit() {
      Gdiplus::GdiplusStartupInput in;
      Gdiplus::GdiplusStartup(&token, &in, nullptr);
    }
    ~GdiInit() {
      if (token) Gdiplus::GdiplusShutdown(token);
    }
  };
  static GdiInit gdi;
  const int n = MultiByteToWideChar(CP_UTF8, 0, path.data(), int(path.size()), nullptr, 0);
  if (n <= 0) return nullptr;
  std::wstring wp(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.data(), int(path.size()), wp.data(), n);
  std::unique_ptr<Gdiplus::Bitmap> bmp(Gdiplus::Bitmap::FromFile(wp.c_str()));
  if (!bmp || bmp->GetLastStatus() != Gdiplus::Ok) return nullptr;
  HICON hicon = nullptr;
  if (bmp->GetHICON(&hicon) != Gdiplus::Ok) return nullptr;
  return hicon;
}

void enable_dark_titlebar(HWND hwnd) {
  if (HMODULE dwm = LoadLibraryW(L"dwmapi.dll")) {
    using Fn = HRESULT(WINAPI*)(HWND, DWORD, LPCVOID, DWORD);
    if (auto fn = reinterpret_cast<Fn>(GetProcAddress(dwm, "DwmSetWindowAttribute"))) {
      BOOL dark = TRUE;
      if (FAILED(fn(hwnd, 20 /*DWMWA_USE_IMMUTABLE_DARK_MODE*/, &dark, sizeof(dark))))
        fn(hwnd, 19, &dark, sizeof(dark));
    }
    FreeLibrary(dwm);
  }
}

class VulkanPresenter final : public Presenter {
 public:
  VulkanPresenter(std::function<void(std::string_view)> log) : log_(std::move(log)) {}
  ~VulkanPresenter() override {
    shutdown();
    if (custom_icon_) DestroyIcon(custom_icon_);
  }

  bool init(const std::string& title, uint32_t w, uint32_t h, const std::string& icon_path) {
    base_title_ = title;
    req_w_.store(w);
    req_h_.store(h);
    if (!create_window(title, w, h, icon_path)) return false;
    return init_vulkan();
  }

  void present(const uint32_t* bgra, uint32_t w, uint32_t h, uint32_t pitch) override;
  bool alive() const override { return !closed_.load(); }
  std::string device_name() const override { return device_name_; }
  void configure(uint32_t w, uint32_t h, int display_mode, int vsync, int upscale_filter, bool fps_counter) override {
    const int old_vsync = vsync_mode_.exchange(vsync);
    upscale_filter_.store(upscale_filter);
    fps_counter_.store(fps_counter);
    if (old_vsync != vsync) resized_ = true;
    if (hwnd_) {
      req_w_.store(w);
      req_h_.store(h);
      req_borderless_.store(display_mode == 1);
      PostMessageW(hwnd_, WM_APP + 10, 0, 0);
    }
  }
  void set_callbacks(std::function<void()> on_settings, std::function<void(bool)> on_focus) override {
    std::lock_guard<std::mutex> lk(cb_m_);
    on_settings_ = std::move(on_settings);
    on_focus_ = std::move(on_focus);
  }

 private:
  void apply_window_mode(HWND hwnd, uint32_t w, uint32_t h, bool borderless) {
    if (borderless) {
      if (!is_borderless_) {
        GetWindowRect(hwnd, &saved_rect_);
        is_borderless_ = true;
      }
      MONITORINFO mi{sizeof(mi)};
      GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi);
      SetWindowLongPtrW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
      SetWindowPos(hwnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                   mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top,
                   SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    } else {
      is_borderless_ = false;
      SetWindowLongPtrW(hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
      RECT r{0, 0, LONG(w), LONG(h)};
      AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
      const int win_w = r.right - r.left, win_h = r.bottom - r.top;
      const int x = saved_rect_.right > saved_rect_.left ? saved_rect_.left : 100;
      const int y = saved_rect_.bottom > saved_rect_.top ? saved_rect_.top : 100;
      SetWindowPos(hwnd, HWND_NOTOPMOST, x, y, win_w, win_h, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    }
    resized_ = true;
  }

  void logf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (log_) log_(buf);
  }

  // ---- window (own thread with a message loop) ----
  static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    auto* self = reinterpret_cast<VulkanPresenter*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    switch (msg) {
      case WM_CLOSE:
        if (self) self->closed_ = true;
        DestroyWindow(hwnd);
        return 0;
      case WM_SIZE:
        if (self) self->resized_ = true;
        return 0;
      case WM_ACTIVATE:
        if (self) {
          const bool active = LOWORD(wp) != WA_INACTIVE;
          std::function<void(bool)> fn;
          { std::lock_guard<std::mutex> lk(self->cb_m_); fn = self->on_focus_; }
          if (fn) fn(active);
        }
        return 0;
      case WM_KEYDOWN:
        {
          // Renderer diagnostics hotkeys: only active when PS3RECOMP_DIAG is set (see the diag launchers).
          static const bool diag_keys = std::getenv("PS3RECOMP_DIAG") != nullptr;
          const bool fresh = !(lp & (1 << 30));  // ignore auto-repeat
          if (diag_keys) {
            if (wp == VK_F4 && fresh) { g_capture_requests.fetch_add(1); return 0; }
            if (wp == VK_F2 && fresh) { g_debug_f2.fetch_add(1); return 0; }
            if (wp == VK_F3 && fresh) { g_debug_f3.fetch_add(1); return 0; }
            if (wp == VK_F5 && fresh) { g_debug_f5.fetch_add(1); return 0; }
            if (wp == VK_F6 && fresh) { g_debug_f6.fetch_add(1); return 0; }
            if (wp == VK_F7 && fresh) { g_debug_f7.fetch_add(1); return 0; }
            if (wp == VK_F11 && fresh) { g_debug_f11.fetch_add(1); return 0; }
            if (wp == VK_F12 && fresh) { g_debug_f12.fetch_add(1); return 0; }
            if (wp == VK_F8) { g_debug_f8.fetch_add(1); return 0; }
            if (wp == VK_F9) { g_debug_f9.fetch_add(1); return 0; }
          }
        }
        if (self && wp == VK_F1) {
          std::function<void()> fn;
          { std::lock_guard<std::mutex> lk(self->cb_m_); fn = self->on_settings_; }
          if (fn) fn();
          return 0;
        }
        break;
      case WM_SYSKEYDOWN:
        if (self && wp == VK_RETURN && (lp & (1 << 29))) {  // Alt+Enter
          const bool next = !self->is_borderless_;
          self->req_borderless_.store(next);
          self->apply_window_mode(hwnd, self->req_w_.load(), self->req_h_.load(), next);
          return 0;
        }
        break;
      case WM_APP + 10:
        if (self) self->apply_window_mode(hwnd, self->req_w_.load(), self->req_h_.load(), self->req_borderless_.load());
        return 0;
      case WM_APP + 11:
        if (self) {
          std::wstring t;
          { std::lock_guard<std::mutex> lk(self->title_m_); t = self->pending_title_; }
          SetWindowTextW(hwnd, t.c_str());
        }
        return 0;
      case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
  }

  bool create_window(const std::string& title, uint32_t w, uint32_t h, const std::string& icon_path) {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    window_thread_ = std::thread([&, title, w, h, icon_path] {
      // If launched from a sandboxed or background wrapper whose thread desktop is a
      // hidden desktop (e.g. exebox-*), attach the presenter thread to the interactive
      // user's Default desktop before making any user32/GDI window calls.
      if (HDESK desk = OpenInputDesktop(0, FALSE, GENERIC_ALL)) {
        SetThreadDesktop(desk);
        CloseDesktop(desk);
      } else if (HDESK def = OpenDesktopW(L"Default", 0, FALSE, GENERIC_ALL)) {
        SetThreadDesktop(def);
        CloseDesktop(def);
      }
      HINSTANCE inst = GetModuleHandleW(nullptr);
      HICON res_icon = LoadIconW(inst, MAKEINTRESOURCEW(101));
      WNDCLASSEXW wc{sizeof(wc)};
      wc.lpfnWndProc = wndproc;
      wc.hInstance = inst;
      wc.hIcon = res_icon;
      wc.hIconSm = res_icon;
      wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
      wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
      wc.lpszClassName = kClass;
      RegisterClassExW(&wc);
      RECT r{0, 0, LONG(w), LONG(h)};
      AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
      const int wn = MultiByteToWideChar(CP_UTF8, 0, title.data(), int(title.size()), nullptr, 0);
      std::wstring wtitle(size_t(std::max(0, wn)), L'\0');
      if (wn > 0) MultiByteToWideChar(CP_UTF8, 0, title.data(), int(title.size()), wtitle.data(), wn);
      const int win_w = r.right - r.left, win_h = r.bottom - r.top;
      HWND hwnd = CreateWindowExW(0, kClass, wtitle.c_str(), WS_OVERLAPPEDWINDOW | WS_VISIBLE, 100, 100,
                                  win_w, win_h, nullptr, nullptr, inst, nullptr);
      if (hwnd) {
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, LONG_PTR(this));
        enable_dark_titlebar(hwnd);
        if (HICON hicon = load_png_icon(icon_path)) {
          custom_icon_ = hicon;
          SendMessageW(hwnd, WM_SETICON, ICON_BIG, LPARAM(hicon));
          SendMessageW(hwnd, WM_SETICON, ICON_SMALL, LPARAM(hicon));
        } else if (res_icon) {
          SendMessageW(hwnd, WM_SETICON, ICON_BIG, LPARAM(res_icon));
          SendMessageW(hwnd, WM_SETICON, ICON_SMALL, LPARAM(res_icon));
        }
        ShowWindow(hwnd, SW_SHOWDEFAULT);  // consumes STARTUPINFO wShowWindow on first call
        ShowWindow(hwnd, SW_RESTORE);
        ShowWindow(hwnd, SW_SHOWNORMAL);   // guarantees visibility even if parent passed SW_HIDE
        SetWindowPos(hwnd, HWND_TOP, 100, 100, win_w, win_h, SWP_SHOWWINDOW);
        UpdateWindow(hwnd);
        SetForegroundWindow(hwnd);
        wchar_t dname[128] = L"?";
        if (HDESK td = GetThreadDesktop(GetCurrentThreadId()))
          GetUserObjectInformationW(td, UOI_NAME, dname, sizeof(dname), nullptr);
        char dname_u8[128] = "?";
        WideCharToMultiByte(CP_UTF8, 0, dname, -1, dname_u8, sizeof(dname_u8), nullptr, nullptr);
        logf("vulkan: window opened on desktop '%s' (hwnd=%p, visible=%d, %dx%d)", dname_u8, (void*)hwnd,
             int(IsWindowVisible(hwnd)), int(w), int(h));
      }
      {
        std::lock_guard<std::mutex> lk(m);
        hwnd_ = hwnd;
        done = true;
      }
      cv.notify_all();
      if (!hwnd) return;
      MSG msg;
      while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
      }
      closed_ = true;
    });
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return done; });
    if (!hwnd_) { logf("vulkan: could not create a window"); return false; }
    return true;
  }

  // ---- Vulkan setup ----
  bool init_vulkan() {
    lib_ = LoadLibraryW(L"vulkan-1.dll");
    if (!lib_) { logf("vulkan: vulkan-1.dll not found (no Vulkan driver installed?)"); return false; }
    vk_.vkGetInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(lib_, "vkGetInstanceProcAddr")));
    if (!vk_.vkGetInstanceProcAddr) return false;
#define LOAD_G(f) vk_.f = reinterpret_cast<PFN_##f>(vk_.vkGetInstanceProcAddr(nullptr, #f));
    VK_GLOBAL_FUNCS(LOAD_G)
#undef LOAD_G

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ps3recomp";
    app.apiVersion = VK_API_VERSION_1_1;
    const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = exts;
    if (vk_.vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) { logf("vulkan: vkCreateInstance failed"); return false; }
#define LOAD_I(f) vk_.f = reinterpret_cast<PFN_##f>(vk_.vkGetInstanceProcAddr(instance_, #f));
    VK_INSTANCE_FUNCS(LOAD_I)
#undef LOAD_I

    VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    sci.hinstance = GetModuleHandleW(nullptr);
    sci.hwnd = hwnd_;
    if (vk_.vkCreateWin32SurfaceKHR(instance_, &sci, nullptr, &surface_) != VK_SUCCESS) { logf("vulkan: surface creation failed"); return false; }

    // Pick a GPU: discrete first (the RTX card on hybrid laptops), then anything that can present.
    uint32_t n = 0;
    vk_.vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    std::vector<VkPhysicalDevice> gpus(n);
    vk_.vkEnumeratePhysicalDevices(instance_, &n, gpus.data());
    int best_score = -1;
    for (VkPhysicalDevice pd : gpus) {
      VkPhysicalDeviceProperties props;
      vk_.vkGetPhysicalDeviceProperties(pd, &props);
      uint32_t qn = 0;
      vk_.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
      std::vector<VkQueueFamilyProperties> qf(qn);
      vk_.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
      for (uint32_t i = 0; i < qn; ++i) {
        VkBool32 present = VK_FALSE;
        vk_.vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface_, &present);
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present) continue;
        const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2
                          : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score > best_score) {
          best_score = score;
          gpu_ = pd;
          queue_family_ = i;
          device_name_ = props.deviceName;
        }
        break;
      }
    }
    if (!gpu_) { logf("vulkan: no GPU can present to the window"); return false; }
    logf("vulkan: using %s", device_name_.c_str());

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* dexts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = dexts;
    if (vk_.vkCreateDevice(gpu_, &dci, nullptr, &device_) != VK_SUCCESS) { logf("vulkan: vkCreateDevice failed"); return false; }
#define LOAD_D(f) vk_.f = reinterpret_cast<PFN_##f>(vk_.vkGetDeviceProcAddr(device_, #f));
    VK_DEVICE_FUNCS(LOAD_D)
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
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vk_.vkCreateSemaphore(device_, &semi, nullptr, &acquired_);
    vk_.vkCreateSemaphore(device_, &semi, nullptr, &rendered_);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vk_.vkCreateFence(device_, &fci, nullptr, &fence_);
    return create_swapchain();
  }

  uint32_t find_memory(uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vk_.vkGetPhysicalDeviceMemoryProperties(gpu_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
      if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
  }

  bool create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vk_.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surface_, &caps);
    extent_ = caps.currentExtent;
    if (extent_.width == UINT32_MAX) {
      RECT r;
      GetClientRect(hwnd_, &r);
      extent_ = {uint32_t(r.right - r.left), uint32_t(r.bottom - r.top)};
    }
    if (extent_.width == 0 || extent_.height == 0) return true;  // minimised; try again later

    uint32_t nf = 0;
    vk_.vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(nf);
    vk_.vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &nf, formats.data());
    VkSurfaceFormatKHR fmt = formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}
                                             : formats[0];
    for (const auto& f : formats)
      if (f.format == VK_FORMAT_B8G8R8A8_UNORM) fmt = f;
    format_ = fmt.format;

    uint32_t nm = 0;
    vk_.vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &nm, nullptr);
    std::vector<VkPresentModeKHR> modes(nm);
    vk_.vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &nm, modes.data());
    auto has_mode = [&](VkPresentModeKHR want) {
      return std::find(modes.begin(), modes.end(), want) != modes.end();
    };
    const int vs = vsync_mode_.load();
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (vs == 0) {
      if (has_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)) mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
      else if (has_mode(VK_PRESENT_MODE_MAILBOX_KHR)) mode = VK_PRESENT_MODE_MAILBOX_KHR;
    } else if (vs == 1) {
      if (has_mode(VK_PRESENT_MODE_MAILBOX_KHR)) mode = VK_PRESENT_MODE_MAILBOX_KHR;
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface_;
    sci.minImageCount = std::max(caps.minImageCount, 2u);
    if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount + 1, caps.maxImageCount);
    sci.imageFormat = fmt.format;
    sci.imageColorSpace = fmt.colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain_;
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    if (vk_.vkCreateSwapchainKHR(device_, &sci, nullptr, &sc) != VK_SUCCESS) { logf("vulkan: swapchain creation failed"); return false; }
    if (swapchain_) vk_.vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = sc;
    uint32_t ni = 0;
    vk_.vkGetSwapchainImagesKHR(device_, swapchain_, &ni, nullptr);
    images_.resize(ni);
    vk_.vkGetSwapchainImagesKHR(device_, swapchain_, &ni, images_.data());
    return true;
  }

  bool ensure_upload(uint32_t w, uint32_t h) {
    if (src_image_ && src_w_ == w && src_h_ == h) return true;
    vk_.vkDeviceWaitIdle(device_);
    destroy_upload();
    const VkDeviceSize size = VkDeviceSize(w) * h * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vk_.vkCreateBuffer(device_, &bci, nullptr, &staging_);
    VkMemoryRequirements mr;
    vk_.vkGetBufferMemoryRequirements(device_, staging_, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) return false;
    vk_.vkAllocateMemory(device_, &mai, nullptr, &staging_mem_);
    vk_.vkBindBufferMemory(device_, staging_, staging_mem_, 0);
    vk_.vkMapMemory(device_, staging_mem_, 0, size, 0, &staging_ptr_);

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vk_.vkCreateImage(device_, &ici, nullptr, &src_image_);
    vk_.vkGetImageMemoryRequirements(device_, src_image_, &mr);
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_.vkAllocateMemory(device_, &mai, nullptr, &src_mem_);
    vk_.vkBindImageMemory(device_, src_image_, src_mem_, 0);
    src_w_ = w;
    src_h_ = h;
    return true;
  }

  void destroy_upload() {
    if (staging_ptr_) vk_.vkUnmapMemory(device_, staging_mem_);
    if (staging_) vk_.vkDestroyBuffer(device_, staging_, nullptr);
    if (staging_mem_) vk_.vkFreeMemory(device_, staging_mem_, nullptr);
    if (src_image_) vk_.vkDestroyImage(device_, src_image_, nullptr);
    if (src_mem_) vk_.vkFreeMemory(device_, src_mem_, nullptr);
    staging_ptr_ = nullptr;
    staging_ = VK_NULL_HANDLE;
    staging_mem_ = src_mem_ = VK_NULL_HANDLE;
    src_image_ = VK_NULL_HANDLE;
  }

  void barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk_.vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
  }

  void shutdown() {
    if (device_) {
      vk_.vkDeviceWaitIdle(device_);
      destroy_upload();
      if (fence_) vk_.vkDestroyFence(device_, fence_, nullptr);
      if (acquired_) vk_.vkDestroySemaphore(device_, acquired_, nullptr);
      if (rendered_) vk_.vkDestroySemaphore(device_, rendered_, nullptr);
      if (pool_) vk_.vkDestroyCommandPool(device_, pool_, nullptr);
      if (swapchain_) vk_.vkDestroySwapchainKHR(device_, swapchain_, nullptr);
      vk_.vkDestroyDevice(device_, nullptr);
      device_ = VK_NULL_HANDLE;
    }
    if (instance_) {
      if (surface_) vk_.vkDestroySurfaceKHR(instance_, surface_, nullptr);
      vk_.vkDestroyInstance(instance_, nullptr);
      instance_ = VK_NULL_HANDLE;
    }
    if (hwnd_ && !closed_) PostMessageW(hwnd_, WM_CLOSE, 0, 0);
    if (window_thread_.joinable()) window_thread_.join();
    if (lib_) FreeLibrary(lib_);
    lib_ = nullptr;
  }

  std::function<void(std::string_view)> log_;
  HICON custom_icon_ = nullptr;
  std::string base_title_;
  std::mutex cb_m_, title_m_;
  std::function<void()> on_settings_;
  std::function<void(bool)> on_focus_;
  std::wstring pending_title_;
  std::atomic<int> vsync_mode_{1};
  std::atomic<int> upscale_filter_{1};
  std::atomic<bool> fps_counter_{true};
  std::atomic<uint32_t> req_w_{1280}, req_h_{720};
  std::atomic<bool> req_borderless_{false};
  bool is_borderless_ = false;
  RECT saved_rect_{100, 100, 1380, 820};
  std::chrono::steady_clock::time_point fps_t0_ = std::chrono::steady_clock::now();
  uint32_t fps_frames_ = 0;
  std::thread window_thread_;
  HWND hwnd_ = nullptr;
  std::atomic<bool> closed_{false}, resized_{false};

  HMODULE lib_ = nullptr;
  Vk vk_;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  std::string device_name_;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkSemaphore acquired_ = VK_NULL_HANDLE, rendered_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D extent_{};
  std::vector<VkImage> images_;

  VkBuffer staging_ = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem_ = VK_NULL_HANDLE, src_mem_ = VK_NULL_HANDLE;
  void* staging_ptr_ = nullptr;
  VkImage src_image_ = VK_NULL_HANDLE;
  uint32_t src_w_ = 0, src_h_ = 0;
};

void VulkanPresenter::present(const uint32_t* bgra, uint32_t w, uint32_t h, uint32_t pitch) {
  if (closed_ || !device_ || !w || !h) return;
  if (resized_.exchange(false) || !swapchain_ || extent_.width == 0) {
    vk_.vkDeviceWaitIdle(device_);
    if (!create_swapchain() || extent_.width == 0) return;
  }
  if (!ensure_upload(w, h)) return;

  vk_.vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  // Upload the frame (row by row: the source pitch may be wider than the width).
  for (uint32_t y = 0; y < h; ++y)
    std::memcpy(static_cast<uint8_t*>(staging_ptr_) + size_t(y) * w * 4, bgra + size_t(y) * pitch, size_t(w) * 4);

  uint32_t index = 0;
  VkResult r = vk_.vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, acquired_, VK_NULL_HANDLE, &index);
  if (r == VK_ERROR_OUT_OF_DATE_KHR) { resized_ = true; return; }
  if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return;
  vk_.vkResetFences(device_, 1, &fence_);

  vk_.vkResetCommandBuffer(cmd_, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_.vkBeginCommandBuffer(cmd_, &bi);

  barrier(src_image_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {w, h, 1};
  vk_.vkCmdCopyBufferToImage(cmd_, staging_, src_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  barrier(src_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  VkImage dst = images_[index];
  barrier(dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
  // Letterbox: keep the frame's aspect ratio inside the window.
  VkClearColorValue black{};
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vk_.vkCmdClearColorImage(cmd_, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
  const double sx = double(extent_.width) / w, sy = double(extent_.height) / h, s = std::min(sx, sy);
  const int32_t dw = int32_t(w * s), dh = int32_t(h * s);
  const int32_t ox = (int32_t(extent_.width) - dw) / 2, oy = (int32_t(extent_.height) - dh) / 2;
  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {int32_t(w), int32_t(h), 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[0] = {ox, oy, 0};
  blit.dstOffsets[1] = {ox + dw, oy + dh, 1};
  const VkFilter filt = upscale_filter_.load() == 0 ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
  vk_.vkCmdBlitImage(cmd_, src_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                     &blit, filt);
  barrier(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
  vk_.vkEndCommandBuffer(cmd_);

  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = 1;
  si.pWaitSemaphores = &acquired_;
  si.pWaitDstStageMask = &wait_stage;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  si.signalSemaphoreCount = 1;
  si.pSignalSemaphores = &rendered_;
  vk_.vkQueueSubmit(queue_, 1, &si, fence_);

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &rendered_;
  pi.swapchainCount = 1;
  pi.pSwapchains = &swapchain_;
  pi.pImageIndices = &index;
  r = vk_.vkQueuePresentKHR(queue_, &pi);
  if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) resized_ = true;

  // Update live FPS counter in window title every 500 ms.
  ++fps_frames_;
  const auto now = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(now - fps_t0_).count();
  if (dt >= 0.5) {
    const double fps = double(fps_frames_) / dt;
    fps_frames_ = 0;
    fps_t0_ = now;
    if (hwnd_) {
      char buf[256];
      if (fps_counter_.load())
        std::snprintf(buf, sizeof(buf), "%s  |  %.1f FPS  [F1: Settings | Alt+Enter: Fullscreen]", base_title_.c_str(), fps);
      else
        std::snprintf(buf, sizeof(buf), "%s", base_title_.c_str());
      const int wn = MultiByteToWideChar(CP_UTF8, 0, buf, -1, nullptr, 0);
      if (wn > 0) {
        std::wstring wt(size_t(wn - 1), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, buf, -1, wt.data(), wn);
        { std::lock_guard<std::mutex> lk(title_m_); pending_title_ = std::move(wt); }
        PostMessageW(hwnd_, WM_APP + 11, 0, 0);
      }
    }
  }
}

}  // namespace

std::unique_ptr<Presenter> create_vulkan_presenter(const std::string& title, std::function<void(std::string_view)> log,
                                                   uint32_t width, uint32_t height, const std::string& icon_path) {
  auto p = std::make_unique<VulkanPresenter>(std::move(log));
  if (!p->init(title, width, height, icon_path)) return nullptr;
  return p;
}

}  // namespace gfx

#else  // !_WIN32

namespace gfx {
std::unique_ptr<Presenter> create_vulkan_presenter(const std::string&, std::function<void(std::string_view)> log,
                                                   uint32_t, uint32_t, const std::string&) {
  if (log) log("vulkan presenter: only implemented for Windows so far");
  return nullptr;
}
}  // namespace gfx

#endif
