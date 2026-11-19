#include "options.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cfg {

std::atomic<bool> g_window_focused{true};
std::atomic<int> g_master_volume{100};
std::atomic<bool> g_mute_in_background{false};
std::atomic<int> g_aniso_level{16};
std::atomic<bool> g_mipmaps_enabled{true};
std::atomic<bool> g_fps_counter_enabled{true};
std::atomic<int> g_fps_cap{60};
std::atomic<int> g_vsync_mode{1};
std::atomic<int> g_upscale_filter{1};
std::atomic<int> g_display_mode{0};
std::atomic<uint32_t> g_window_width{1280};
std::atomic<uint32_t> g_window_height{720};
std::atomic<uint32_t> g_settings_gen{1};

namespace {

std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  return s.substr(i);
}

std::string to_lower_ascii(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

}  // namespace

bool Options::is_mod_disabled(const std::string& filename) const {
  const std::string target = to_lower_ascii(filename);
  for (const auto& m : disabled_mods) {
    if (to_lower_ascii(m) == target) return true;
  }
  return false;
}

void Options::set_mod_disabled(const std::string& filename, bool disabled) {
  const std::string target = to_lower_ascii(filename);
  disabled_mods.erase(std::remove_if(disabled_mods.begin(), disabled_mods.end(),
                                     [&](const std::string& m) { return to_lower_ascii(m) == target; }),
                      disabled_mods.end());
  if (disabled && !filename.empty()) disabled_mods.push_back(filename);
}

void Options::apply_preset(int preset) {
  quality_preset = std::clamp(preset, 0, 4);
  switch (quality_preset) {
    case 0:  // Low / Handheld
      window_width = 1280; window_height = 720;
      anisotropic_filtering = 1; mipmaps = false; upscale_filter = 1; vsync = 0; fps_cap = 60;
      break;
    case 1:  // Balanced
      window_width = 1280; window_height = 720;
      anisotropic_filtering = 4; mipmaps = true; upscale_filter = 1; vsync = 1; fps_cap = 60;
      break;
    case 2:  // High (Recommended)
      window_width = 1920; window_height = 1080;
      anisotropic_filtering = 16; mipmaps = true; upscale_filter = 1; vsync = 1; fps_cap = 60;
      break;
    case 3:  // Ultra (1440p)
      window_width = 2560; window_height = 1440;
      anisotropic_filtering = 16; mipmaps = true; upscale_filter = 1; vsync = 1; fps_cap = 60;
      break;
    default:
      break;
  }
}

Options& get() {
  static Options o;
  return o;
}

std::filesystem::path& ini_path() {
  static std::filesystem::path p;
  return p;
}

void sync_atomics(const Options& o) {
  g_master_volume.store(std::clamp(o.volume, 0, 100), std::memory_order_relaxed);
  g_mute_in_background.store(o.mute_in_background, std::memory_order_relaxed);
  g_aniso_level.store(std::clamp(o.anisotropic_filtering, 1, 16), std::memory_order_relaxed);
  g_mipmaps_enabled.store(o.mipmaps, std::memory_order_relaxed);
  g_fps_counter_enabled.store(o.fps_counter, std::memory_order_relaxed);
  g_fps_cap.store(std::max(0, o.fps_cap), std::memory_order_relaxed);
  g_vsync_mode.store(std::clamp(o.vsync, 0, 2), std::memory_order_relaxed);
  g_upscale_filter.store(std::clamp(o.upscale_filter, 0, 1), std::memory_order_relaxed);
  g_display_mode.store(std::clamp(o.display_mode, 0, 1), std::memory_order_relaxed);
  g_window_width.store(std::clamp<uint32_t>(o.window_width, 640, 7680), std::memory_order_relaxed);
  g_window_height.store(std::clamp<uint32_t>(o.window_height, 360, 4320), std::memory_order_relaxed);
  g_settings_gen.fetch_add(1, std::memory_order_release);
}

bool load(const std::filesystem::path& file, Options& out) {
  std::ifstream f(file);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = trim(line.substr(0, eq));
    const std::string v = trim(line.substr(eq + 1));
    const int iv = std::atoi(v.c_str());
    if (k == "quality_preset") out.quality_preset = std::clamp(iv, 0, 4);
    else if (k == "window_width") out.window_width = uint32_t(std::clamp(iv, 640, 7680));
    else if (k == "window_height") out.window_height = uint32_t(std::clamp(iv, 360, 4320));
    else if (k == "display_mode") out.display_mode = std::clamp(iv, 0, 1);
    else if (k == "vsync") out.vsync = std::clamp(iv, 0, 2);
    else if (k == "fps_cap") out.fps_cap = std::max(0, iv);
    else if (k == "anisotropic_filtering") out.anisotropic_filtering = std::clamp(iv, 1, 16);
    else if (k == "mipmaps") out.mipmaps = iv != 0;
    else if (k == "upscale_filter") out.upscale_filter = std::clamp(iv, 0, 1);
    else if (k == "fps_counter") out.fps_counter = iv != 0;
    else if (k == "volume") out.volume = std::clamp(iv, 0, 100);
    else if (k == "mute_in_background") out.mute_in_background = iv != 0;
    else if (k == "disabled_mods") {
      out.disabled_mods.clear();
      std::stringstream ss(v);
      std::string item;
      while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) out.disabled_mods.push_back(item);
      }
    }
  }
  return true;
}

bool save(const std::filesystem::path& file, const Options& o) {
  std::error_code ec;
  if (!file.parent_path().empty()) std::filesystem::create_directories(file.parent_path(), ec);
  std::ofstream f(file);
  if (!f) return false;
  f << "# CellForge / Skate3-VK Graphics & Audio Options\n";
  f << "# Press F1 in-game (or click Settings in CellForge) to edit.\n";
  f << "quality_preset=" << o.quality_preset << "\n";
  f << "window_width=" << o.window_width << "\n";
  f << "window_height=" << o.window_height << "\n";
  f << "display_mode=" << o.display_mode << "\n";
  f << "vsync=" << o.vsync << "\n";
  f << "fps_cap=" << o.fps_cap << "\n";
  f << "anisotropic_filtering=" << o.anisotropic_filtering << "\n";
  f << "mipmaps=" << (o.mipmaps ? 1 : 0) << "\n";
  f << "upscale_filter=" << o.upscale_filter << "\n";
  f << "fps_counter=" << (o.fps_counter ? 1 : 0) << "\n";
  f << "volume=" << o.volume << "\n";
  f << "mute_in_background=" << (o.mute_in_background ? 1 : 0) << "\n";
  f << "disabled_mods=";
  for (size_t i = 0; i < o.disabled_mods.size(); ++i) {
    if (i) f << ",";
    f << o.disabled_mods[i];
  }
  f << "\n";
  return f.good();
}

}  // namespace cfg
