#pragma once
// ============================================================================
// CellForge Mod SDK — Single-Header C++20 Modding API (v1)
// ============================================================================
// Build your mod as a Windows x64 DLL and drop it into the game's `mods/`
// folder (e.g. %LOCALAPPDATA%\Games\Skate3-VK\mods\).
//
// CellForge automatically loads every *.dll in `mods/` on startup and calls:
//
//   CELLFORGE_MOD_ENTRY(api) {
//     cellforge::init(api);
//     cellforge::logf("MyMod loaded! Guest memory base = %p", api->guest_base);
//     return 0;  // 0 = success, non-zero = abort load
//   }
//
// ABI compatibility: matches CellForge's src/runtime/ppu_runtime.h and
// src/runtime/ppu_mod_api.h (PS3RECOMP_MOD_API_VERSION == 1).
// ============================================================================

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#define CF_BSWAP16(x) _byteswap_ushort(x)
#define CF_BSWAP32(x) _byteswap_ulong(x)
#define CF_BSWAP64(x) _byteswap_uint64(x)
#else
#define CF_BSWAP16(x) __builtin_bswap16(x)
#define CF_BSWAP32(x) __builtin_bswap32(x)
#define CF_BSWAP64(x) __builtin_bswap64(x)
#endif

// ----------------------------------------------------------------------------
// Core CellForge PPU ABI (must stay layout-identical to ppu_runtime.h)
// ----------------------------------------------------------------------------
namespace ppu {

struct alignas(16) v128 {
  uint8_t b[16];
};

struct CRField {
  uint8_t lt = 0, gt = 0, eq = 0, so = 0;
};

struct Context {
  uint64_t r[32]{};          // General-purpose registers (r3 = return / arg0, r4..r10 = arg1..arg7)
  double f[32]{};            // Floating-point registers (f1 = float return / fp arg0)
  v128 v[32]{};              // AltiVec/VMX 128-bit vector registers
  CRField cr[8]{};           // Condition register fields cr0..cr7
  uint64_t lr = 0;           // Link register
  uint64_t ctr = 0;          // Count register
  uint8_t xer_so = 0, xer_ov = 0, xer_ca = 0;
  uint8_t xer_bc = 0;
  uint32_t fpscr = 0;
  uint32_t vscr = 0x10000;
  uint32_t vrsave = 0;
  uint32_t res_addr = 0;
  uint64_t res_val = 0;
  bool res_valid = false;
  uint8_t* base = nullptr;   // Start of the 4 GiB PS3 guest address space
};

using Fn = void (*)(Context&);

}  // namespace ppu

#define PS3RECOMP_MOD_API_VERSION 1

struct Ps3RecompModApi {
  uint32_t version;
  uint8_t* guest_base;                           // Guest effective address 0x00000000
  ppu::Fn (*hook)(uint32_t addr, ppu::Fn fn);    // Replace guest fn at `addr`; returns previous fn
  void (*unhook)(uint32_t addr);                 // Restore original recompiled function
  void (*call)(ppu::Context& c, uint32_t addr);  // Call any guest function by PS3 address
  void (*log)(const char* msg);                  // Write a message to the CellForge runtime log
  // Added after v1 (check for nullptr on very old runtimes): code address of an imported HLE function
  // (module e.g. "sys_net", name or "nid:0xXXXXXXXX") to pass to hook(). 0 if the game doesn't import it.
  uint32_t (*hle_code)(const char* module, const char* name);
  // Permanent zero-initialised guest memory (never freed) for strings/structs/buffers handed to the game. 0 on failure.
  uint32_t (*guest_alloc)(uint32_t size, uint32_t align);
  // Guest function descriptor (OPD) whose code is the host function `fn` (for vtables / callbacks). 0 on failure.
  uint32_t (*make_opd)(ppu::Fn fn);
};

#if defined(_WIN32)
#define PS3RECOMP_MOD_EXPORT extern "C" __declspec(dllexport)
#else
#define PS3RECOMP_MOD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Convenience macro for defining the mod entry point.
#define CELLFORGE_MOD_ENTRY(api_param) \
  PS3RECOMP_MOD_EXPORT int ps3recomp_mod_init(const Ps3RecompModApi* api_param)

// ----------------------------------------------------------------------------
// High-Level C++ Helpers (`cellforge::`)
// ----------------------------------------------------------------------------
namespace cellforge {

inline const Ps3RecompModApi* g_api = nullptr;

// Call once at the start of ps3recomp_mod_init. Returns false on version mismatch.
inline bool init(const Ps3RecompModApi* api) {
  if (!api || api->version != PS3RECOMP_MOD_API_VERSION) return false;
  g_api = api;
  return true;
}

inline uint8_t* guest_base() { return g_api ? g_api->guest_base : nullptr; }

// ---- Logging ---------------------------------------------------------------
inline void log(const char* msg) {
  if (g_api && g_api->log && msg) g_api->log(msg);
}

inline void logf(const char* fmt, ...) {
  if (!g_api || !g_api->log || !fmt) return;
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_api->log(buf);
}

// ---- Function Hooking & Calling --------------------------------------------
inline ppu::Fn hook(uint32_t guest_addr, ppu::Fn replacement) {
  return (g_api && g_api->hook) ? g_api->hook(guest_addr, replacement) : nullptr;
}

inline void unhook(uint32_t guest_addr) {
  if (g_api && g_api->unhook) g_api->unhook(guest_addr);
}

inline void call(ppu::Context& ctx, uint32_t guest_addr) {
  if (g_api && g_api->call) g_api->call(ctx, guest_addr);
}

// RAII hook guard that stores the original function pointer and can unhook automatically.
class Hook {
 public:
  Hook() = default;
  bool install(uint32_t addr, ppu::Fn fn) {
    addr_ = addr;
    orig_ = hook(addr, fn);
    return orig_ != nullptr;
  }
  void remove() {
    if (orig_) {
      unhook(addr_);
      orig_ = nullptr;
    }
  }
  void call_original(ppu::Context& c) const {
    if (orig_) orig_(c);
  }
  ppu::Fn original() const { return orig_; }
  uint32_t address() const { return addr_; }
  explicit operator bool() const { return orig_ != nullptr; }

 private:
  uint32_t addr_ = 0;
  ppu::Fn orig_ = nullptr;
};

// Hooks an imported system-library (HLE) function such as sys_net::socket. Returns the previous implementation
// (the runtime's built-in one or the "unimplemented import" stub), or nullptr if the game doesn't import it.
inline ppu::Fn hook_import(const char* module, const char* name, ppu::Fn replacement) {
  if (!g_api || !g_api->hle_code) return nullptr;
  const uint32_t code = g_api->hle_code(module, name);
  return code ? g_api->hook(code, replacement) : nullptr;
}

inline uint32_t make_opd(ppu::Fn fn) { return (g_api && g_api->make_opd) ? g_api->make_opd(fn) : 0; }

inline uint32_t guest_alloc(uint32_t size, uint32_t align = 16) {
  return (g_api && g_api->guest_alloc) ? g_api->guest_alloc(size, align) : 0;
}

// ---- Big-Endian PS3 Guest Memory Read / Write ------------------------------
inline uint8_t read_u8(uint32_t ea) { return guest_base()[ea]; }
inline uint16_t read_u16(uint32_t ea) {
  uint16_t v;
  std::memcpy(&v, guest_base() + ea, 2);
  return CF_BSWAP16(v);
}
inline uint32_t read_u32(uint32_t ea) {
  uint32_t v;
  std::memcpy(&v, guest_base() + ea, 4);
  return CF_BSWAP32(v);
}
inline uint64_t read_u64(uint32_t ea) {
  uint64_t v;
  std::memcpy(&v, guest_base() + ea, 8);
  return CF_BSWAP64(v);
}
inline int32_t read_i32(uint32_t ea) { return static_cast<int32_t>(read_u32(ea)); }
inline float read_f32(uint32_t ea) {
  const uint32_t u = read_u32(ea);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline double read_f64(uint32_t ea) {
  const uint64_t u = read_u64(ea);
  double d;
  std::memcpy(&d, &u, 8);
  return d;
}

inline void write_u8(uint32_t ea, uint8_t v) { guest_base()[ea] = v; }
inline void write_u16(uint32_t ea, uint16_t v) {
  v = CF_BSWAP16(v);
  std::memcpy(guest_base() + ea, &v, 2);
}
inline void write_u32(uint32_t ea, uint32_t v) {
  v = CF_BSWAP32(v);
  std::memcpy(guest_base() + ea, &v, 4);
}
inline void write_u64(uint32_t ea, uint64_t v) {
  v = CF_BSWAP64(v);
  std::memcpy(guest_base() + ea, &v, 8);
}
inline void write_i32(uint32_t ea, int32_t v) { write_u32(ea, static_cast<uint32_t>(v)); }
inline void write_f32(uint32_t ea, float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  write_u32(ea, u);
}
inline void write_f64(uint32_t ea, double d) {
  uint64_t u;
  std::memcpy(&u, &d, 8);
  write_u64(ea, u);
}

inline std::string read_cstr(uint32_t ea, size_t max_len = 256) {
  const char* p = reinterpret_cast<const char*>(guest_base() + ea);
  size_t n = 0;
  while (n < max_len && p[n] != '\0') ++n;
  return std::string(p, n);
}

inline void read_bytes(uint32_t ea, void* dst, size_t len) {
  std::memcpy(dst, guest_base() + ea, len);
}

inline void write_bytes(uint32_t ea, const void* src, size_t len) {
  std::memcpy(guest_base() + ea, src, len);
}

// ---- Byte Pattern Scanner --------------------------------------------------
// Scans guest memory [start_ea, start_ea + length) for a hex byte pattern with
// "??" wildcards, e.g. "38 60 ?? ?? 4E 80 00 20". Returns guest EA or 0.
inline uint32_t find_pattern(uint32_t start_ea, uint32_t length, std::string_view pattern) {
  std::vector<uint8_t> bytes;
  std::vector<bool> mask;
  size_t i = 0;
  while (i < pattern.size()) {
    while (i < pattern.size() && (pattern[i] == ' ' || pattern[i] == '\t')) ++i;
    if (i >= pattern.size()) break;
    if (pattern[i] == '?') {
      bytes.push_back(0);
      mask.push_back(false);
      while (i < pattern.size() && pattern[i] == '?') ++i;
    } else if (i + 1 < pattern.size()) {
      char hex[3] = {pattern[i], pattern[i + 1], '\0'};
      bytes.push_back(static_cast<uint8_t>(std::strtoul(hex, nullptr, 16)));
      mask.push_back(true);
      i += 2;
    } else {
      break;
    }
  }
  if (bytes.empty() || length < bytes.size()) return 0;
  const uint8_t* mem = guest_base() + start_ea;
  const size_t limit = length - bytes.size();
  for (size_t off = 0; off <= limit; ++off) {
    bool match = true;
    for (size_t j = 0; j < bytes.size(); ++j) {
      if (mask[j] && mem[off + j] != bytes[j]) {
        match = false;
        break;
      }
    }
    if (match) return start_ea + static_cast<uint32_t>(off);
  }
  return 0;
}

// ---- Lightweight INI Config Loader -----------------------------------------
// Lets mods read simple `key=value` settings from `mods/<my_mod>.ini`.
class Config {
 public:
  bool load(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      line = trim(line);
      if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') continue;
      const size_t eq = line.find('=');
      if (eq == std::string::npos) continue;
      values_[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return true;
  }

  std::string get_string(const std::string& key, const std::string& def = {}) const {
    auto it = values_.find(key);
    return it != values_.end() ? it->second : def;
  }

  int get_int(const std::string& key, int def = 0) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    return std::atoi(it->second.c_str());
  }

  float get_float(const std::string& key, float def = 0.0f) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    return static_cast<float>(std::atof(it->second.c_str()));
  }

  bool get_bool(const std::string& key, bool def = false) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    const std::string& v = it->second;
    if (v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "FALSE" || v == "no" || v == "off") return false;
    return def;
  }

 private:
  static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
  }
  std::unordered_map<std::string, std::string> values_;
};

}  // namespace cellforge
