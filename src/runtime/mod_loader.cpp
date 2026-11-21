#include "ppu_mod_api.h"
#include "hle.h"
#include "lv2.h"
#include "options.h"

#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#endif

namespace ppu {
namespace {

void (*g_log)(const char*) = nullptr;

void log_msg(const std::string& s) {
  if (g_log) g_log(s.c_str());
}

Fn api_hook(uint32_t addr, Fn fn) { return hook_function(addr, fn); }
void api_unhook(uint32_t addr) { unhook_function(addr); }
void api_call(Context& c, uint32_t addr) { call_indirect(c, addr); }
void api_log(const char* msg) { log_msg(std::string("[mod] ") + msg); }
uint32_t api_guest_alloc(uint32_t size, uint32_t align) {
  const uint32_t a = lv2::Kernel::get().hle.alloc(size, align ? align : 16);
  if (a) std::memset(lv2::Kernel::get().base() + a, 0, size);
  return a;
}
uint32_t api_make_opd(Fn fn) { return hle::make_opd(fn); }
uint32_t api_hle_code(const char* module, const char* name) { return hle::import_code(module ? module : "", name ? name : ""); }

using InitFn = int (*)(const Ps3RecompModApi*);

std::vector<std::string> list_libraries(const std::string& dir) {
  std::vector<std::string> out;
#if defined(_WIN32)
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA((dir + "\\*.dll").c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) return out;
  do {
    if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) out.push_back(dir + "\\" + fd.cFileName);
  } while (FindNextFileA(h, &fd));
  FindClose(h);
#else
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 3 && n.compare(n.size() - 3, 3, ".so") == 0) out.push_back(dir + "/" + n);
    }
    closedir(d);
  }
#endif
  return out;
}

InitFn open_library(const std::string& path, std::string& err) {
#if defined(_WIN32)
  HMODULE m = LoadLibraryA(path.c_str());
  if (!m) { err = "LoadLibrary failed (" + std::to_string(GetLastError()) + ")"; return nullptr; }
  auto f = reinterpret_cast<InitFn>(reinterpret_cast<void*>(GetProcAddress(m, "ps3recomp_mod_init")));
  if (!f) { err = "missing export ps3recomp_mod_init"; FreeLibrary(m); }
  return f;
#else
  void* m = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!m) { err = dlerror(); return nullptr; }
  auto f = reinterpret_cast<InitFn>(dlsym(m, "ps3recomp_mod_init"));
  if (!f) { err = "missing export ps3recomp_mod_init"; dlclose(m); }
  return f;
#endif
}

std::mutex g_mods_mu;
std::vector<ModStatus> g_mod_statuses;

}  // namespace

std::vector<ModStatus> get_mod_status_list() {
  std::lock_guard<std::mutex> lk(g_mods_mu);
  return g_mod_statuses;
}

int load_mods(const char* dir, uint8_t* guest_base, void (*log)(const char*)) {
  g_log = log;
  static Ps3RecompModApi api;
  api = Ps3RecompModApi{PS3RECOMP_MOD_API_VERSION, guest_base, api_hook, api_unhook, api_call, api_log, api_hle_code, api_guest_alloc, api_make_opd};

  std::vector<ModStatus> statuses;
  int loaded = 0;
  for (const std::string& path : list_libraries(dir)) {
    ModStatus st;
    st.path = path;
    st.filename = std::filesystem::path(path).filename().string();

    if (cfg::get().is_mod_disabled(st.filename)) {
      st.disabled = true;
      st.detail = "disabled in Settings";
      log_msg("mod " + st.filename + ": skipped (disabled in Settings)");
      statuses.push_back(std::move(st));
      continue;
    }

    std::string err;
    InitFn init = open_library(path, err);
    if (!init) {
      st.detail = err;
      log_msg("mod " + path + ": " + err);
      statuses.push_back(std::move(st));
      continue;
    }
    const int rc = init(&api);
    if (rc != 0) {
      st.detail = "init returned " + std::to_string(rc);
      log_msg("mod " + path + ": " + st.detail);
      statuses.push_back(std::move(st));
      continue;
    }
    st.loaded = true;
    st.detail = "active";
    log_msg("loaded mod " + path);
    statuses.push_back(std::move(st));
    ++loaded;
  }

  {
    std::lock_guard<std::mutex> lk(g_mods_mu);
    g_mod_statuses = std::move(statuses);
  }
  return loaded;
}

}  // namespace ppu
