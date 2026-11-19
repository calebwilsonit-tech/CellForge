#include "ppu_mod_api.h"

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

}  // namespace

int load_mods(const char* dir, uint8_t* guest_base, void (*log)(const char*)) {
  g_log = log;
  static Ps3RecompModApi api;
  api = Ps3RecompModApi{PS3RECOMP_MOD_API_VERSION, guest_base, api_hook, api_unhook, api_call, api_log};

  int loaded = 0;
  for (const std::string& path : list_libraries(dir)) {
    std::string err;
    InitFn init = open_library(path, err);
    if (!init) { log_msg("mod " + path + ": " + err); continue; }
    const int rc = init(&api);
    if (rc != 0) { log_msg("mod " + path + ": init returned " + std::to_string(rc)); continue; }
    log_msg("loaded mod " + path);
    ++loaded;
  }
  return loaded;
}

}  // namespace ppu
