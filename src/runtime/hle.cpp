#define _CRT_SECURE_NO_WARNINGS
#include "hle.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

#include "lv2.h"
#include "nid.h"

namespace hle {
namespace {

struct Entry {
  std::string name;
  ppu::Fn fn = nullptr;
};

struct Unimpl {
  std::string module, name;
  uint32_t nid;
  bool reported = false;
};

std::map<std::string, std::unordered_map<uint32_t, Entry>>& registry() {
  static std::map<std::string, std::unordered_map<uint32_t, Entry>> r;
  return r;
}

std::mutex g_m;
std::unordered_map<uint32_t, Unimpl> g_unimpl;            // code address -> import
std::map<std::pair<std::string, uint32_t>, uint32_t> g_opds;  // (module, nid) -> OPD
uint32_t g_code_keys = 0, g_code_used = 0;
constexpr uint32_t kCodeKeySpace = 0x10000;

// Called for every import that has no implementation. The import stub put the
// target code address in CTR right before `bctr`, which identifies the import.
void unimplemented_stub(ppu::Context& c) {
  lv2::Kernel& k = lv2::Kernel::get();
  std::string what = "unknown import";
  {
    std::lock_guard<std::mutex> lk(g_m);
    auto it = g_unimpl.find(uint32_t(c.ctr));
    if (it != g_unimpl.end()) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "%s::%s (NID 0x%08X)", it->second.module.c_str(),
                    it->second.name.empty() ? "?" : it->second.name.c_str(), it->second.nid);
      what = buf;
      if (it->second.reported && !k.strict) { c.r[3] = 0; return; }
      it->second.reported = true;
    }
  }
  if (k.strict) throw std::runtime_error("unimplemented import " + what);
  k.logf("unimplemented import called: %s (returning CELL_OK)", what.c_str());
  c.r[3] = 0;
}

// PS3RECOMP_HLE_TRACE=<sec>: from <sec> seconds after boot, log implemented
// import calls (first 600, with args and result), then per-function counts.
struct Traced {
  std::string module, name;
  ppu::Fn fn;
  uint64_t count = 0;
};
std::unordered_map<uint32_t, Traced> g_traced;  // code address -> import
double g_trace_start = -1;

double since_boot() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void traced_stub(ppu::Context& c) {
  auto it = g_traced.find(uint32_t(c.ctr));
  if (it == g_traced.end()) { c.r[3] = 0; return; }
  Traced& t = it->second;
  if (since_boot() < g_trace_start) { t.fn(c); return; }
  static std::atomic<uint32_t> logged{0};
  const uint64_t a3 = c.r[3], a4 = c.r[4], a5 = c.r[5], lr = c.lr;
  const std::string& who = lv2::Kernel::get().current().name;
  t.fn(c);
  {
    std::lock_guard<std::mutex> lk(g_m);
    ++t.count;
  }
  if (logged++ < 600)
    lv2::Kernel::get().logf("hle %s::%s(0x%llX, 0x%llX, 0x%llX) = 0x%llX  lr=0x%08llX [%s]", t.module.c_str(), t.name.c_str(),
                            (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5,
                            (unsigned long long)c.r[3], (unsigned long long)lr, who.c_str());
  static double last_report = 0;
  const double now = since_boot();
  bool report = false;
  {
    std::lock_guard<std::mutex> lk(g_m);
    if (now - last_report > 10) { last_report = now; report = true; }
  }
  if (report) {
    std::string s;
    std::lock_guard<std::mutex> lk(g_m);
    for (auto& [code, x] : g_traced)
      if (x.count) s += "\n    " + x.module + "::" + x.name + " " + std::to_string(x.count);
    lv2::Kernel::get().logf("hle call counts since trace start:%s", s.c_str());
  }
}

uint32_t next_code_key() {
  lv2::Kernel& k = lv2::Kernel::get();
  if (!g_code_keys || g_code_used + 4 > kCodeKeySpace) {
    g_code_keys = k.hle.alloc(kCodeKeySpace, 0x1000);
    g_code_used = 0;
    // Fill with `trap` so stray native execution of these addresses is obvious.
    for (uint32_t i = 0; i < kCodeKeySpace; i += 4) ppu::st32(k.base(), g_code_keys + i, 0x7FE00008);
  }
  const uint32_t a = g_code_keys + g_code_used;
  g_code_used += 4;
  return a;
}

}  // namespace

void register_function(const char* module, const char* name, ppu::Fn fn) {
  registry()[module][nid::of(name)] = Entry{name, fn};
}

namespace {
// Module + NID first; NIDs are derived from globally unique names, so fall back to
// any module when the game imports a function under a module name we filed differently.
const Entry* find_entry(const std::string& module, uint32_t n) {
  auto& reg = registry();
  auto m = reg.find(module);
  if (m != reg.end()) {
    auto it = m->second.find(n);
    if (it != m->second.end()) return &it->second;
  }
  for (auto& [name, funcs] : reg) {
    auto it = funcs.find(n);
    if (it != funcs.end()) return &it->second;
  }
  return nullptr;
}
}  // namespace

uint32_t make_opd(ppu::Fn fn) {
  lv2::Kernel& k = lv2::Kernel::get();
  uint32_t code;
  {
    std::lock_guard<std::mutex> lk(g_m);
    code = next_code_key();
  }
  const uint32_t opd = k.hle.alloc(8, 8);
  ppu::st32(k.base(), opd, code);
  ppu::st32(k.base(), opd + 4, 0);
  ppu::register_function(code, fn, nullptr);
  return opd;
}

ppu::Fn lookup(const std::string& module, const std::string& name) {
  const Entry* e = find_entry(module, nid::of(name));
  return e ? e->fn : nullptr;
}

ResolveStats resolve_imports(const std::vector<elf::ImportModule>& imports) {
  lv2::Kernel& k = lv2::Kernel::get();
  ResolveStats st;
  for (const auto& m : imports) {
    ++st.modules;
    for (const auto& f : m.funcs) {
      ++st.functions;
      uint32_t opd = 0;
      {
        std::lock_guard<std::mutex> lk(g_m);
        auto it = g_opds.find({m.name, f.nid});
        if (it != g_opds.end()) opd = it->second;
      }
      if (!opd) {
        const uint32_t code = next_code_key();
        opd = k.hle.alloc(8, 8);
        ppu::st32(k.base(), opd, code);
        ppu::st32(k.base(), opd + 4, 0);  // HLE code doesn't use r2

        const Entry* e = find_entry(m.name, f.nid);
        static const char* trace_env = std::getenv("PS3RECOMP_HLE_TRACE");
        if (e && trace_env) {
          since_boot();
          g_trace_start = std::atof(trace_env);
          std::lock_guard<std::mutex> lk(g_m);
          g_traced[code] = Traced{m.name, e->name, e->fn};
          ppu::register_function(code, traced_stub, nullptr);
          ++st.implemented;
        } else if (e) {
          ppu::register_function(code, e->fn, nullptr);
          ++st.implemented;
        } else {
          std::lock_guard<std::mutex> lk(g_m);
          g_unimpl[code] = Unimpl{m.name, nid::name_of(f.nid), f.nid};
          ppu::register_function(code, unimplemented_stub, nullptr);
          ++st.stubbed;
        }
        std::lock_guard<std::mutex> lk(g_m);
        g_opds[{m.name, f.nid}] = opd;
      }
      if (f.slot) ppu::st32(k.base(), f.slot, opd);
    }
    for (const auto& v : m.vars) {
      ++st.variables;
      k.logf("imported variable %s::0x%08X (%s) is not supported", m.name.c_str(), v.nid, nid::name_of(v.nid).c_str());
    }
  }
  return st;
}

uint32_t import_opd(const std::string& module, const std::string& name) {
  std::lock_guard<std::mutex> lk(g_m);
  auto it = g_opds.find({module, nid::of(name)});
  return it == g_opds.end() ? 0 : it->second;
}

uint32_t import_code(const std::string& module, const std::string& name) {
  const uint32_t n = name.rfind("nid:", 0) == 0 ? uint32_t(std::strtoul(name.c_str() + 4, nullptr, 16)) : nid::of(name);
  uint32_t opd;
  {
    std::lock_guard<std::mutex> lk(g_m);
    auto it = g_opds.find({module, n});
    if (it == g_opds.end()) return 0;
    opd = it->second;
  }
  return ppu::ld32(lv2::Kernel::get().base(), opd);
}

// ---- printf -------------------------------------------------------------------
std::string format_guest(ppu::Context& c, uint32_t fmt_addr, int gpr, int fpr) {
  lv2::Kernel& k = lv2::Kernel::get();
  const std::string fmt = k.read_str(fmt_addr, 8192);
  // Variadic arguments: r3..r10, then the parameter save area at r1 + 48.
  auto next_int = [&]() -> uint64_t {
    const int i = gpr++;
    if (i <= 10) return c.r[i];
    return ppu::ld64(c.base, uint32_t(c.r[1] + 48 + 8 * (i - 3)));
  };
  auto next_double = [&]() -> double {
    const uint64_t bits = next_int();  // the GPR/stack slot is consumed too
    if (fpr <= 13) return c.f[fpr++];
    return std::bit_cast<double>(bits);
  };

  std::string out;
  for (size_t i = 0; i < fmt.size(); ++i) {
    if (fmt[i] != '%') { out += fmt[i]; continue; }
    size_t j = i + 1;
    std::string spec = "%";
    while (j < fmt.size() && std::strchr("-+ #0", fmt[j])) spec += fmt[j++];
    auto num = [&]() {
      if (j < fmt.size() && fmt[j] == '*') { spec += std::to_string(int32_t(next_int())); ++j; return; }
      while (j < fmt.size() && std::isdigit(uint8_t(fmt[j]))) spec += fmt[j++];
    };
    num();
    if (j < fmt.size() && fmt[j] == '.') { spec += fmt[j++]; num(); }
    int longs = 0;
    bool half = false;
    while (j < fmt.size() && std::strchr("hlLqjzt", fmt[j])) {
      if (fmt[j] == 'h') half = true; else ++longs;
      ++j;
    }
    if (j >= fmt.size()) break;
    const char conv = fmt[j];
    char buf[512];
    buf[0] = 0;
    switch (conv) {
      case '%': out += '%'; break;
      case 'd': case 'i': {
        const uint64_t v = next_int();
        const long long s = longs ? (long long)v : half ? (long long)(int16_t)v : (long long)(int32_t)v;
        std::snprintf(buf, sizeof buf, (spec + "lld").c_str(), s);
        out += buf;
        break;
      }
      case 'u': case 'x': case 'X': case 'o': {
        const uint64_t v = next_int();
        const unsigned long long u = longs ? v : half ? uint16_t(v) : uint32_t(v);
        std::snprintf(buf, sizeof buf, (spec + "ll" + conv).c_str(), u);
        out += buf;
        break;
      }
      case 'c': out += char(next_int()); break;
      case 'p': std::snprintf(buf, sizeof buf, "0x%08X", uint32_t(next_int())); out += buf; break;
      case 's': {
        const std::string s = k.read_str(uint32_t(next_int()), 4096);
        std::snprintf(buf, sizeof buf, (spec + "s").c_str(), s.c_str());
        out += buf;
        break;
      }
      case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A':
        std::snprintf(buf, sizeof buf, (spec + conv).c_str(), next_double());
        out += buf;
        break;
      default: out += spec + conv; break;
    }
    i = j;
  }
  return out;
}

}  // namespace hle
