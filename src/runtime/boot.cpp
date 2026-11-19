#define _CRT_SECURE_NO_WARNINGS
#include "boot.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>

#include "elf64.h"
#include "guest_memory.h"
#include "hle.h"
#include "lv2.h"
#include "ppu_mod_api.h"
#include "ppu_runtime.h"
#include "ps3_exec.h"
#include "rsx.h"
#include "vfs.h"

#include <filesystem>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace ppu {

namespace {
std::function<void(std::string_view)> g_log;
void log_c(const char* s) { if (g_log) g_log(s); }

#if defined(_WIN32)
// Last-chance report for host crashes in guest code: which guest address was
// touched, where in ps3game.exe it happened (look the offset up in ps3game.map),
// and the guest thread's registers and call trace.
LONG WINAPI crash_filter(EXCEPTION_POINTERS* ep) {
  static std::atomic<bool> once{false};
  if (once.exchange(true)) return EXCEPTION_CONTINUE_SEARCH;
  lv2::Kernel& k = lv2::Kernel::get();
  const EXCEPTION_RECORD* er = ep->ExceptionRecord;
  const uintptr_t rip = uintptr_t(er->ExceptionAddress);
  const uintptr_t module = uintptr_t(GetModuleHandleW(nullptr));
  k.logf("host exception 0x%08lX at ps3game.exe+0x%llX", er->ExceptionCode, (unsigned long long)(rip - module));
  if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
    const uintptr_t addr = uintptr_t(er->ExceptionInformation[1]);
    const uintptr_t base = uintptr_t(k.base());
    const char* kind = er->ExceptionInformation[0] == 1 ? "write" : er->ExceptionInformation[0] == 8 ? "execute" : "read";
    if (addr >= base && addr - base < (uintptr_t(1) << 32))
      k.logf("  guest %s of unmapped address 0x%08llX", kind, (unsigned long long)(addr - base));
    else
      k.logf("  host %s of 0x%llX (outside guest memory)", kind, (unsigned long long)addr);
  }
  // Host call stack: walk the faulting thread's frames from the exception context.
  {
    CONTEXT ctx = *ep->ContextRecord;
    std::string trace;
    for (int depth = 0; depth < 24 && ctx.Rip; ++depth) {
      HMODULE mod = nullptr;
      char name[MAX_PATH] = "?";
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(ctx.Rip), &mod))
        GetModuleFileNameA(mod, name, sizeof name);
      const char* base_name = std::strrchr(name, '\\');
      char frame[160];
      std::snprintf(frame, sizeof frame, "\n    %s+0x%llX", base_name ? base_name + 1 : name,
                    (unsigned long long)(ctx.Rip - uintptr_t(mod)));
      trace += frame;
      DWORD64 image_base = 0;
      PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
      if (!fn) {  // leaf function: return address is at the top of the stack
        ctx.Rip = *reinterpret_cast<DWORD64*>(ctx.Rsp);
        ctx.Rsp += 8;
        continue;
      }
      void* handler_data = nullptr;
      DWORD64 establisher = 0;
      RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, fn, &ctx, &handler_data, &establisher, nullptr);
    }
    k.logf("  host call stack (ps3game.exe offsets -> ps3game.map):%s", trace.c_str());
  }
  lv2::Thread& t = k.current();
  k.logf("  in guest thread '%s'", t.name.c_str());
  k.dump_context(t.ctx);
  std::fflush(nullptr);
  return EXCEPTION_CONTINUE_SEARCH;
}

// PS3RECOMP_HWWATCH=<hex guest address>: hardware write watchpoint (DR0) on the
// primary thread. Each write to that word logs the writing host instruction
// (ps3game.exe offset -> ps3game.map) and the guest LR.
LONG WINAPI hwwatch_handler(EXCEPTION_POINTERS* ep) {
  if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || !(ep->ContextRecord->Dr6 & 1))
    return EXCEPTION_CONTINUE_SEARCH;
  static int hits = 0;
  lv2::Kernel& k = lv2::Kernel::get();
  const uintptr_t module = uintptr_t(GetModuleHandleW(nullptr));
  const uintptr_t addr = uintptr_t(ep->ContextRecord->Dr0);
  if (hits++ < 32)
    k.logf("hwwatch: [0x%08llX] = 0x%08X written at ps3game.exe+0x%llX, guest lr=0x%llX",
           (unsigned long long)(addr - uintptr_t(k.base())), k.r32(uint32_t(addr - uintptr_t(k.base()))),
           (unsigned long long)(uintptr_t(ep->ExceptionRecord->ExceptionAddress) - module),
           (unsigned long long)k.current().ctx.lr);
  ep->ContextRecord->Dr6 = 0;
  return EXCEPTION_CONTINUE_EXECUTION;
}
void arm_hwwatch(uint8_t* host) {
  AddVectoredExceptionHandler(1, hwwatch_handler);
  CONTEXT ctx{};
  ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
  ctx.Dr0 = uintptr_t(host);
  ctx.Dr7 = 1 | (1ull << 16) | (3ull << 18);  // L0, break on write, 4 bytes
  SetThreadContext(GetCurrentThread(), &ctx);
}
#endif
}  // namespace

int boot(const BootOptions& opt) {
#if defined(_WIN32)
  // Games sleep and poll in short intervals (sys_timer_usleep, flip/label waits);
  // Windows' default 15.6 ms timer tick turns each of those into a long stall.
  timeBeginPeriod(1);
  SetUnhandledExceptionFilter(crash_filter);
#endif
  lv2::Kernel& k = lv2::Kernel::get();
  k.log = opt.log ? opt.log : [](std::string_view s) {
    std::fprintf(stderr, "[ps3] %.*s\n", int(s.size()), s.data());
  };
  k.tty = opt.tty;
  k.strict = opt.strict;
  g_log = k.log;

  elf::LoadResult lr = elf::load_file(opt.elf_path);
  if (!lr.ok) { k.logf("cannot load %s: %s", opt.elf_path.c_str(), lr.error.c_str()); return -1; }
  const elf::Image& img = lr.image;

  // The guest address space lives for the rest of the process.
  static GuestMemory* mem = new GuestMemory();
  if (!mem->ok()) { k.logf("cannot reserve the 4 GiB guest address space"); return -1; }
  for (const auto& s : img.segments) {
    if (!mem->map(uint32_t(s.vaddr), s.memsz)) { k.logf("cannot map segment at 0x%llX", (unsigned long long)s.vaddr); return -1; }
    std::memcpy(mem->base() + s.vaddr, s.data.data(), s.data.size());
  }
  k.init(mem);

  // Virtual file system.
  std::filesystem::path resolved_game_root;
  {
    namespace fs = std::filesystem;
    fs::path root = opt.game_root;
    if (root.empty()) {
      const fs::path dir = fs::absolute(opt.elf_path).parent_path();
      root = (dir.filename() == "USRDIR" && dir.parent_path().filename() == "PS3_GAME")
                 ? dir.parent_path().parent_path()
                 : dir;
    }
    resolved_game_root = root;
    vfs::mount_standard(root, opt.data_root.empty() ? fs::path("ps3_data") : fs::path(opt.data_root));
    const auto& sfo = vfs::game_sfo().strings;
    auto title = sfo.find("TITLE"), id = sfo.find("TITLE_ID");
    k.logf("game root %s%s%s%s", root.string().c_str(), title != sfo.end() ? " - " : "",
           title != sfo.end() ? title->second.c_str() : "", id != sfo.end() ? (" [" + id->second + "]").c_str() : "");
  }

  rsx::Rsx::get().use_gpu = opt.gpu_render;
  // Graphics: open the window immediately on startup so the user sees it during boot/loading.
  if (!opt.headless && !rsx::Rsx::get().presenter_factory) {
    const auto& sfo = vfs::game_sfo().strings;
    auto title_it = sfo.find("TITLE");
    auto id_it = sfo.find("TITLE_ID");
    std::string title = opt.window_title;
    if (title.empty()) {
      if (id_it != sfo.end() && (id_it->second == "BLUS30464" || id_it->second == "BLES00760"))
        title = "Skate3-VK";
      else if (title_it != sfo.end())
        title = title_it->second + " - CellForge";
      else
        title = "CellForge";
    }
    std::string icon_path = opt.icon_path;
    if (icon_path.empty() && !resolved_game_root.empty()) {
      std::error_code ec;
      const auto candidate = resolved_game_root / "PS3_GAME" / "ICON0.PNG";
      if (std::filesystem::exists(candidate, ec)) icon_path = candidate.string();
    }
    rsx::Rsx::get().presenter_factory = [title, icon_path, log = k.log]() -> std::shared_ptr<gfx::Presenter> {
      return std::shared_ptr<gfx::Presenter>(gfx::create_vulkan_presenter(title, log, 1280, 720, icon_path));
    };
    rsx::Rsx::get().set_presenter(rsx::Rsx::get().presenter_factory());
  }

  const elf::ProcParam pp = elf::read_proc_param(img);
  if (pp.present) k.sdk_version = pp.sdk_version;
  const elf::TlsInfo tls = elf::read_tls(img);
  k.set_tls_image(tls.vaddr, tls.filesz, tls.memsz);

  // Recompiled code + system libraries.
  if (function_table().empty()) ppu_register_functions();
  hle::register_builtin_modules();
  std::vector<elf::ImportModule> imports;
  std::string err;
  if (!elf::read_imports(img, imports, err)) { k.logf("import table: %s", err.c_str()); return -1; }
  const hle::ResolveStats st = hle::resolve_imports(imports);
  k.logf("imports: %u modules, %u functions (%u implemented, %u stubbed), %u variables", st.modules, st.functions,
         st.implemented, st.stubbed, st.variables);

  if (!opt.mods_dir.empty()) {
    const int n = load_mods(opt.mods_dir.c_str(), mem->base(), log_c);
    k.logf("%d mod(s) loaded", n);
  }

  // argv / envp: the kernel passes arrays of 64-bit pointers; the game's startup
  // code compacts them in place into 32-bit pointer arrays before main().
  std::vector<std::string> argv{"/app_home/EBOOT.BIN"};
  argv.insert(argv.end(), opt.args.begin(), opt.args.end());
  const uint32_t argv_arr = k.hle.alloc(uint32_t(8 * (argv.size() + 1)), 16);
  for (size_t i = 0; i < argv.size(); ++i) k.w64(argv_arr + uint32_t(8 * i), k.alloc_str(argv[i]));
  k.w64(argv_arr + uint32_t(8 * argv.size()), 0);
  const uint32_t envp_arr = k.hle.alloc(8, 16);
  k.w64(envp_arr, 0);

  // Primary thread. Entry is an OPD.
  auto main = k.create_thread(uint32_t(img.entry), 0, pp.primary_prio, pp.primary_stacksize, true, "main");
  if (!main) { k.logf("cannot allocate the primary thread"); return -1; }
  Context& c = main->ctx;
  c.r[3] = argv.size();
  c.r[4] = argv_arr;
  c.r[5] = envp_arr;
  c.r[6] = 0;
  c.r[7] = main->id;
  c.r[8] = tls.vaddr;
  c.r[9] = tls.filesz;
  c.r[10] = tls.memsz;
  c.r[12] = pp.malloc_pagesize;

  int status = 0;
  k.start_thread_dumper();
#if defined(_WIN32)
  if (const char* w = std::getenv("PS3RECOMP_HWWATCH")) arm_hwwatch(k.base() + std::strtoul(w, nullptr, 16));
#endif
  try {
    k.run_here(main);
    status = int(int32_t(main->ctx.r[3]));
    k.logf("primary thread returned %d", status);
  } catch (const lv2::ProcessExit& e) {
    status = e.status;
    k.logf("process exited with status %d", status);
  } catch (const lv2::ThreadExit& e) {
    status = int(int32_t(e.value));
    k.logf("primary thread exited with %d", status);
  } catch (const std::exception& e) {
    k.logf("fatal: %s", e.what());
    k.dump_context(main->ctx);
    status = -2;
  }
  k.detach_all_threads();
  return status;
}

}  // namespace ppu
