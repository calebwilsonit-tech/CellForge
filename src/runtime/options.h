#pragma once
// User-configurable graphics, display, and audio options (options.ini).
// Shared between the game runtime (Skate3-VK.exe, F1 settings menu) and CellForge-Setup.exe.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace cfg {

struct Options {
  int quality_preset = 2;          // 0=Low/Handheld, 1=Balanced, 2=High, 3=Ultra, 4=Custom
  uint32_t window_width = 1280;    // 1280, 1600, 1920, 2560, 3840
  uint32_t window_height = 720;    // 720, 900, 1080, 1440, 2160
  int display_mode = 0;            // 0=Windowed, 1=Borderless Fullscreen
  int vsync = 1;                   // 0=Off (Immediate), 1=Fast (Mailbox), 2=On (FIFO)
  int fps_cap = 60;                // 0=Uncapped, 30, 60, 120
  int anisotropic_filtering = 16;  // 1, 2, 4, 8, 16
  bool mipmaps = true;             // GPU mipmap chain + trilinear filtering
  int upscale_filter = 1;          // 0=Nearest, 1=Linear
  bool fps_counter = true;         // Show live FPS in window title
  int volume = 100;                // 0..100
  bool mute_in_background = false; // Mute audio when game window is unfocused

  void apply_preset(int preset);
};

// Global active options + live change notification (called when user clicks Apply).
Options& get();
std::filesystem::path& ini_path();

bool load(const std::filesystem::path& file, Options& out);
bool save(const std::filesystem::path& file, const Options& in);

// Runtime state shared across subsystems (lock-free hot-path reads).
extern std::atomic<bool> g_window_focused;
extern std::atomic<int> g_master_volume;         // 0..100
extern std::atomic<bool> g_mute_in_background;
extern std::atomic<int> g_aniso_level;           // 1..16
extern std::atomic<bool> g_mipmaps_enabled;
extern std::atomic<bool> g_fps_counter_enabled;
extern std::atomic<int> g_fps_cap;               // 0, 30, 60, 120
extern std::atomic<int> g_vsync_mode;            // 0, 1, 2
extern std::atomic<int> g_upscale_filter;        // 0, 1
extern std::atomic<int> g_display_mode;          // 0=Windowed, 1=Borderless
extern std::atomic<uint32_t> g_window_width;
extern std::atomic<uint32_t> g_window_height;
extern std::atomic<uint32_t> g_settings_gen;     // incremented on Apply so presenter/renderer react

void sync_atomics(const Options& o);

}  // namespace cfg
