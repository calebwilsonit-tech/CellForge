// Boots the test ELF like a game (imports, TLS, threads, syscalls), then exercises
// the LV2 kernel objects directly from several host threads.
// usage: ps3_boot_test <test.elf>

#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <cstring>
#include <filesystem>
#include <fstream>

#include <cmath>

#include "audio.h"
#include "boot.h"
#include "hle.h"
#include "input.h"
#include "lv2.h"
#include "rsx.h"
#include "vfs.h"
#include "nid.h"
#include "test_layout.h"

namespace {

int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) {
  ++g_checks;
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++g_fail;
}

// Calls a syscall the way guest code does.
ppu::Context sc(uint32_t n, std::initializer_list<uint64_t> args) {
  ppu::Context c;
  c.base = lv2::Kernel::get().base();
  int i = 3;
  for (uint64_t a : args) c.r[i++] = a;
  c.r[11] = n;
  lv2::Kernel::get().syscall(c);
  return c;
}

void test_nid() {
  std::printf("nid\n");
  const auto d = nid::sha1("abc", 3);
  check(d[0] == 0xA9 && d[1] == 0x99 && d[19] == 0x9D, "sha1(\"abc\")");
  // Reference values from the PS3 homebrew/emulation community's NID tables.
  check(nid::of("cellSysutilCheckCallback") == 0x189A74DA, "nid(cellSysutilCheckCallback) == 0x189A74DA");
  check(nid::of("sys_initialize_tls") == 0x744680A2, "nid(sys_initialize_tls) == 0x744680A2");
  check(nid::of("sys_time_get_system_time") == 0x8461E528, "nid(sys_time_get_system_time) == 0x8461E528");
  check(nid::of("cellPadInit") == 0x1CF98800, "nid(cellPadInit) == 0x1CF98800");
  check(nid::name_of(nid::of("cellGcmSetFlipMode")) == "cellGcmSetFlipMode", "name database lookup");
}

void test_boot(const char* elf) {
  std::printf("boot\n");
  std::string tty, log;
  ppu::BootOptions opt;
  opt.elf_path = elf;
  opt.data_root = (std::filesystem::temp_directory_path() / "ps3recomp_boot_data").string();
  opt.headless = true;
  opt.tty = [&](std::string_view s) { tty += s; };
  opt.log = [&](std::string_view s) { log += s; log += '\n'; };
  const int status = ppu::boot(opt);
  std::printf("  --- runtime log ---\n%s  -------------------\n", log.c_str());

  lv2::Kernel& k = lv2::Kernel::get();
  const uint32_t R = test_layout::kResults;
  check(status == 0, "sys_process_exit(0) ends boot with status 0");
  check(k.r32(R + 0) == 0x12345678, "TLS image visible through r13 after sys_initialize_tls");
  check(k.r32(R + 4) != 0, "sys_time_get_system_time (HLE) returned a time");
  check(k.r32(R + 8) == 0, "unimplemented import returns CELL_OK");
  check(log.find("cellFoo::? (NID 0xDEADBEEF)") != std::string::npos, "unimplemented import is logged with module + NID");
  check(k.r32(R + 12) == 7, "thread created via sys_ppu_thread_create joined (syscall 44) with exit value 7");
  check(k.r32(R + 16) == 2000, "two threads x 1000 lwarx/stwcx. increments == 2000");
  check(k.r32(R + 20) == 15, "_sys_printf returned the formatted length");
  check(tty.find("hello guest 42\n") != std::string::npos, "_sys_printf formatted guest varargs");
  check(tty.find("tty ok\n") != std::string::npos, "sys_tty_write (syscall 403)");
  check(log.find("imports: 2 modules, 5 functions (4 implemented, 1 stubbed)") != std::string::npos,
        "import resolution summary");
  // The callbacks above capture locals of this function: detach them before it returns.
  k.log = [](std::string_view s) { std::printf("  [log] %.*s\n", int(s.size()), s.data()); };
  k.tty = nullptr;
}

void test_kernel() {
  using namespace lv2;
  Kernel& k = Kernel::get();
  std::printf("kernel objects\n");
  const uint32_t scratch = k.heap.alloc(0x1000, 16);
  check(scratch != 0, "heap allocation");

  // mutex: contention between 4 host threads
  sc(100, {scratch, 0});
  const uint32_t mtx = k.r32(scratch);
  std::atomic<int> inside{0}, max_inside{0};
  int counter = 0;
  {
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
      ts.emplace_back([&] {
        for (int i = 0; i < 2000; ++i) {
          sc(102, {mtx, 0});
          const int n = ++inside;
          int m = max_inside.load();
          while (n > m && !max_inside.compare_exchange_weak(m, n)) {}
          ++counter;
          --inside;
          sc(104, {mtx});
        }
      });
    for (auto& t : ts) t.join();
  }
  check(counter == 8000 && max_inside == 1, "sys_mutex gives mutual exclusion");
  check(sc(104, {mtx}).r[3] == CELL_EPERM, "unlock by non-owner -> EPERM");
  check(sc(102, {mtx, 0}).r[3] == CELL_OK && sc(102, {mtx, 0}).r[3] == CELL_EDEADLK, "relock non-recursive -> EDEADLK");
  check(sc(103, {mtx}).r[3] == CELL_EDEADLK, "trylock by owner -> EDEADLK");

  // cond: producer/consumer
  sc(105, {scratch + 4, mtx, 0});
  const uint32_t cnd = k.r32(scratch + 4);
  bool ready = false;
  std::thread consumer([&] {
    sc(102, {mtx, 0});
    while (!ready) sc(107, {cnd, 0});
    sc(104, {mtx});
  });
  sc(104, {mtx});  // release the lock held by this thread from the EDEADLK test
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  sc(102, {mtx, 0});
  ready = true;
  sc(108, {cnd});
  sc(104, {mtx});
  consumer.join();
  check(true, "sys_cond wait/signal wakes the consumer");
  sc(102, {mtx, 0});
  check(sc(107, {cnd, 1000}).r[3] == CELL_ETIMEDOUT, "cond wait times out");
  sc(104, {mtx});
  check(sc(101, {mtx}).r[3] == CELL_EPERM, "destroying a mutex bound to a cond -> EPERM");

  // semaphore
  sc(90, {scratch + 8, 0, 0, 2});
  const uint32_t sem = k.r32(scratch + 8);
  check(sc(93, {sem}).r[3] == CELL_EBUSY, "semaphore trywait on 0 -> EBUSY");
  check(sc(94, {sem, 2}).r[3] == CELL_OK && sc(94, {sem, 1}).r[3] == CELL_EBUSY, "semaphore post respects max");
  check(sc(92, {sem, 0}).r[3] == CELL_OK, "semaphore wait");
  sc(114, {sem, scratch + 12});
  check(k.r32(scratch + 12) == 1, "semaphore get_value");

  // event queue + port across threads
  sc(128, {scratch + 16, 0, 0, 16});
  const uint32_t q = k.r32(scratch + 16);
  sc(134, {scratch + 20, 1, 0xABCD});
  const uint32_t port = k.r32(scratch + 20);
  check(sc(138, {port, 1, 2, 3}).r[3] == CELL_ENOTCONN, "send on unconnected port -> ENOTCONN");
  sc(136, {port, q});
  ppu::Context got;
  std::thread rx([&] { got = sc(130, {q, 0, 0}); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  sc(138, {port, 11, 22, 33});
  rx.join();
  check(got.r[3] == CELL_OK && got.r[4] == 0xABCD && got.r[5] == 11 && got.r[6] == 22 && got.r[7] == 33,
        "event delivered in r4..r7 to a blocked receiver");
  check(sc(130, {q, 0, 1000}).r[3] == CELL_ETIMEDOUT, "event receive times out");

  // memory
  sc(348, {0x10000, 0x200, scratch + 24});
  const uint32_t blk = k.r32(scratch + 24);
  check(blk >= kHeapBase && (blk & 0xFFFF) == 0, "sys_memory_allocate (64K pages)");
  check(sc(349, {blk}).r[3] == CELL_OK && sc(349, {blk}).r[3] == CELL_EINVAL, "sys_memory_free, double free rejected");
  check(sc(147, {}).r[3] == kTimebaseFrequency, "timebase frequency");
}

// Calls a registered HLE function the way an import stub would.
ppu::Context call(const char* module, const char* name, std::initializer_list<uint64_t> args) {
  ppu::Context c;
  c.base = lv2::Kernel::get().base();
  int i = 3;
  for (uint64_t a : args) c.r[i++] = a;
  ppu::Fn fn = hle::lookup(module, name);
  if (!fn) { std::printf("  FAIL  no HLE function %s\n", name); ++g_fail; return c; }
  fn(c);
  return c;
}

// Minimal PARAM.SFO writer (little-endian PSF).
void write_sfo(const std::filesystem::path& p) {
  struct E { std::string key; bool is_int; std::string s; uint32_t i; };
  const std::vector<E> es = {{"PARENTAL_LEVEL", true, "", 3}, {"TITLE", false, "Test Game", 0}, {"TITLE_ID", false, "TEST00001", 0}};
  std::vector<uint8_t> keys, data, out;
  auto le32 = [](std::vector<uint8_t>& v, uint32_t x) { for (int k = 0; k < 4; ++k) v.push_back(uint8_t(x >> (8 * k))); };
  auto le16 = [](std::vector<uint8_t>& v, uint16_t x) { v.push_back(uint8_t(x)); v.push_back(uint8_t(x >> 8)); };
  std::vector<uint8_t> index;
  for (const auto& e : es) {
    le16(index, uint16_t(keys.size()));
    le16(index, e.is_int ? 0x0404 : 0x0204);
    const uint32_t len = e.is_int ? 4 : uint32_t(e.s.size() + 1);
    le32(index, len);
    le32(index, len);
    le32(index, uint32_t(data.size()));
    keys.insert(keys.end(), e.key.begin(), e.key.end());
    keys.push_back(0);
    if (e.is_int) le32(data, e.i);
    else { data.insert(data.end(), e.s.begin(), e.s.end()); data.push_back(0); }
  }
  while (keys.size() % 4) keys.push_back(0);
  le32(out, 0x46535000);
  le32(out, 0x101);
  le32(out, uint32_t(0x14 + index.size()));
  le32(out, uint32_t(0x14 + index.size() + keys.size()));
  le32(out, uint32_t(es.size()));
  out.insert(out.end(), index.begin(), index.end());
  out.insert(out.end(), keys.begin(), keys.end());
  out.insert(out.end(), data.begin(), data.end());
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
}

void test_fs_game_pad() {
  namespace fs = std::filesystem;
  using namespace lv2;
  Kernel& k = Kernel::get();
  const fs::path tmp = fs::temp_directory_path() / "ps3recomp_fs_test";
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp / "game" / "PS3_GAME" / "USRDIR");
  write_sfo(tmp / "game" / "PS3_GAME" / "PARAM.SFO");
  std::ofstream(tmp / "game" / "PS3_GAME" / "USRDIR" / "data.txt") << "0123456789";
  vfs::mount_standard(tmp / "game", tmp / "data");

  const uint32_t g = k.heap.alloc(0x2000, 16);
  auto str = [&](uint32_t off, const char* s) { std::memcpy(k.base() + g + off, s, std::strlen(s) + 1); return g + off; };
  const uint32_t fdp = g + 0x1000, n = g + 0x1008, buf = g + 0x1100, st = g + 0x1200, pos = g + 0x1010;

  std::printf("cellFs\n");
  check(call("sys_fs", "cellFsOpen", {str(0, "/app_home/data.txt"), 0, fdp, 0, 0}).r[3] == 0, "open /app_home file");
  const uint32_t fd = k.r32(fdp);
  call("sys_fs", "cellFsRead", {fd, buf, 4, n});
  check(k.r64(n) == 4 && std::memcmp(k.base() + buf, "0123", 4) == 0, "read");
  call("sys_fs", "cellFsLseek", {fd, uint64_t(-3), 2, pos});
  check(k.r64(pos) == 7, "lseek from end");
  call("sys_fs", "cellFsRead", {fd, buf, 10, n});
  check(k.r64(n) == 3 && std::memcmp(k.base() + buf, "789", 3) == 0, "read to EOF");
  call("sys_fs", "cellFsFstat", {fd, st});
  check(k.r64(st + 36) == 10 && (k.r32(st) & 0170000) == 0100000, "fstat size + regular-file mode");
  check(call("sys_fs", "cellFsClose", {fd}).r[3] == 0 && call("sys_fs", "cellFsClose", {fd}).r[3] == 0x8001002A,
        "close; double close -> EBADF");
  check(call("sys_fs", "cellFsOpen", {str(0, "/dev_bdvd/PS3_GAME/x.txt"), 0101, fdp, 0, 0}).r[3] == CELL_EPERM,
        "disc is read-only");
  check(call("sys_fs", "cellFsOpen", {str(0, "/dev_hdd0/../../../escape.txt"), 0101, fdp, 0, 0}).r[3] == CELL_ENOENT,
        "path traversal out of a mount is rejected");
  check(call("sys_fs", "cellFsOpen", {str(0, "/dev_nope/x"), 0, fdp, 0, 0}).r[3] == CELL_ENOENT, "unknown device");

  check(call("sys_fs", "cellFsMkdir", {str(0, "/dev_hdd0/game/TEST"), 0777}).r[3] == 0, "mkdir");
  check(call("sys_fs", "cellFsMkdir", {str(0, "/dev_hdd0/game/TEST"), 0777}).r[3] == 0x80010014, "mkdir existing -> EEXIST");
  check(call("sys_fs", "cellFsOpen", {str(0, "/dev_hdd0/game/TEST/save.dat"), 0102 | 01000, fdp, 0, 0}).r[3] == 0,
        "create + truncate for read/write");
  const uint32_t wfd = k.r32(fdp);
  std::memcpy(k.base() + buf, "hello", 5);
  call("sys_fs", "cellFsWrite", {wfd, buf, 5, n});
  std::memcpy(k.base() + buf, "J", 1);
  call("sys_fs", "cellFsWriteWithOffset", {wfd, 0, buf, 1, n});
  call("sys_fs", "cellFsClose", {wfd});
  std::ifstream in(tmp / "data" / "dev_hdd0" / "game" / "TEST" / "save.dat");
  std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();
  check(contents == "Jello", "write + write-with-offset reach the host file");
  call("sys_fs", "cellFsStat", {str(0, "/dev_hdd0/game/TEST/save.dat"), st});
  check(k.r64(st + 36) == 5, "stat");

  call("sys_fs", "cellFsOpendir", {str(0, "/dev_hdd0/game/TEST"), fdp});
  const uint32_t dfd = k.r32(fdp);
  std::vector<std::string> names;
  for (int i = 0; i < 10; ++i) {
    call("sys_fs", "cellFsReaddir", {dfd, buf, n});
    if (k.r64(n) == 0) break;
    names.emplace_back(reinterpret_cast<const char*>(k.base() + buf + 2), k.base()[buf + 1]);
  }
  call("sys_fs", "cellFsClosedir", {dfd});
  check(names.size() == 3 && names[2] == "save.dat" && k.base()[buf] == 2, "opendir/readdir lists ., .., save.dat");
  check(call("sys_fs", "cellFsRmdir", {str(0, "/dev_hdd0/game/TEST")}).r[3] == 0x8001003C, "rmdir non-empty -> ENOTEMPTY");
  check(call("sys_fs", "cellFsRename", {str(0, "/dev_hdd0/game/TEST/save.dat"), str(0x100, "/dev_hdd0/game/TEST/b.dat")}).r[3] == 0 &&
            call("sys_fs", "cellFsUnlink", {str(0, "/dev_hdd0/game/TEST/b.dat")}).r[3] == 0 &&
            call("sys_fs", "cellFsRmdir", {str(0, "/dev_hdd0/game/TEST")}).r[3] == 0,
        "rename, unlink, rmdir");

  std::printf("cellGame\n");
  call("cellGame", "cellGameGetParamString", {100, buf, 32});
  check(std::string(reinterpret_cast<const char*>(k.base() + buf)) == "TEST00001", "TITLE_ID from PARAM.SFO");
  call("cellGame", "cellGameGetParamString", {0, buf, 64});
  check(std::string(reinterpret_cast<const char*>(k.base() + buf)) == "Test Game", "TITLE from PARAM.SFO");
  call("cellGame", "cellGameGetParamInt", {102, n});
  check(k.r32(n) == 3, "PARENTAL_LEVEL int param");
  call("cellGame", "cellGameContentPermit", {buf, buf + 128});
  check(std::string(reinterpret_cast<const char*>(k.base() + buf + 128)) == "/dev_bdvd/PS3_GAME/USRDIR", "content permit paths");

  std::printf("cellPad\n");
  input::set_provider([](int port, input::PadState& s) {
    if (port != 0) return false;
    s.connected = true;
    s.digital2 = input::CROSS;
    s.digital1 = input::START;
    s.lx = 0;
    return true;
  });
  check(call("sys_io", "cellPadGetData", {0, buf}).r[3] == 0x80121103, "GetData before init -> UNINITIALIZED");
  check(call("sys_io", "cellPadInit", {1}).r[3] == 0, "init");
  check(call("sys_io", "cellPadInit", {1}).r[3] == 0x80121101, "double init -> ALREADY_INITIALIZED");
  call("sys_io", "cellPadGetInfo2", {buf});
  check(k.r32(buf) == 1 && k.r32(buf + 4) == 1 && (k.r32(buf + 12) & 1), "GetInfo2: one pad connected");
  check(call("sys_io", "cellPadGetData", {0, buf}).r[3] == 0, "GetData");
  auto btn = [&](int i) { return ppu::ld16(k.base(), buf + 4 + 2 * i); };
  check(k.r32(buf) == 24 && btn(2) == input::START && btn(3) == input::CROSS, "digital buttons (start + cross)");
  check(btn(6) == 0 && btn(7) == 0x80 && btn(14) == 255, "left stick + cross pressure");
  check(call("sys_io", "cellPadGetData", {1, buf}).r[3] == 0x80121104, "port beyond max_connect -> INVALID_PARAMETER");
  call("sys_io", "cellPadEnd", {});
  input::set_provider(nullptr);
  fs::remove_all(tmp, ec);
}

uint32_t g_ctx = 0;                              // cellGcm context used by the draw tests
std::shared_ptr<gfx::Presenter> g_capture;       // capture presenter installed by test_gcm
std::atomic<int> g_flip_handler_calls{0};
void test_flip_handler(ppu::Context& c) {
  if (c.r[3] == 1) ++g_flip_handler_calls;
}

template <class Pred>
bool wait_until(Pred p, int ms = 2000) {
  for (int i = 0; i < ms; ++i) {
    if (p()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return p();
}

void test_gcm() {
  using namespace lv2;
  Kernel& k = Kernel::get();
  rsx::Rsx& r = rsx::Rsx::get();
  auto capture = std::make_shared<gfx::CapturePresenter>();
  g_capture = capture;
  r.set_presenter(capture);

  std::printf("cellGcm / RSX\n");
  // 1 MB-aligned IO area for the command buffer.
  const uint32_t io_area = k.heap.alloc(0x200000, 0x100000);
  const uint32_t scratch = k.heap.alloc(0x1000, 16);
  check(call("cellGcmSys", "_cellGcmInitBody", {scratch, 0x40000, 0x100000, io_area}).r[3] == 0, "_cellGcmInitBody");
  const uint32_t ctx = k.r32(scratch);
  g_ctx = ctx;
  check(k.r32(ctx) == io_area + 0x1000 && k.r32(ctx + 8) == io_area + 0x1000 && k.r32(ctx + 12) != 0,
        "context begins 4 KB into IO memory and has a callback");

  call("cellGcmSys", "cellGcmGetConfiguration", {scratch + 0x100});
  check(k.r32(scratch + 0x100) == 0xC0000000 && k.r32(scratch + 0x104) == io_area, "configuration");
  call("cellGcmSys", "cellGcmAddressToOffset", {io_area + 0x1234, scratch + 0x200});
  check(k.r32(scratch + 0x200) == 0x1234, "address -> IO offset");
  call("cellGcmSys", "cellGcmAddressToOffset", {0xC0001000, scratch + 0x200});
  check(k.r32(scratch + 0x200) == 0x1000, "address -> local offset");
  const uint32_t extra = k.heap.alloc(0x100000, 0x100000);
  call("cellGcmSys", "cellGcmMapMainMemory", {extra, 0x100000, scratch + 0x204});
  const uint32_t extra_io = k.r32(scratch + 0x204);
  call("cellGcmSys", "cellGcmGetOffsetTable", {scratch + 0x208});
  const uint32_t io_table = k.r32(scratch + 0x208);
  check(extra_io == 0x100000 && ppu::ld16(k.base(), io_table + 2 * (extra >> 20)) == 1,
        "MapMainMemory picks the next IO page and updates the guest offset table");

  const uint32_t fb_offset = 0x100000, pitch = 1280 * 4;
  call("cellGcmSys", "cellGcmSetDisplayBuffer", {0, fb_offset, pitch, 1280, 720});
  const uint32_t fh = hle::make_opd(test_flip_handler);
  call("cellGcmSys", "cellGcmSetFlipHandler", {fh});
  call("cellGcmSys", "cellGcmResetFlipStatus", {});

  // Build a command stream the way a game's inline libgcm code would.
  uint32_t cur = k.r32(ctx + 8);
  auto emit = [&](uint32_t method, std::initializer_list<uint32_t> args) {
    ppu::st32(k.base(), cur, (uint32_t(args.size()) << 18) | method);
    cur += 4;
    for (uint32_t a : args) { ppu::st32(k.base(), cur, a); cur += 4; }
  };
  emit(0x0194, {0xFEED0000});                 // colour A in local memory
  emit(0x0200, {1280u << 16, 720u << 16});    // clip horizontal + vertical
  emit(0x020C, {pitch});                      // pitch A
  emit(0x0210, {fb_offset});                  // colour A offset
  emit(0x1D90, {0xFF336699});                 // clear colour (ARGB)
  emit(0x1D94, {0xF0});                       // clear RGBA
  emit(0x0064, {5 * 16});                     // semaphore offset -> label 5
  emit(0x006C, {0x77});                       // release
  emit(0x0050, {0x55});                       // reference
  k.w32(ctx + 8, cur);
  call("cellGcmSys", "cellGcmSetFlipCommand", {ctx, 0});
  const uint32_t ctrl = k.r32(ctx) ? r.control_addr() : 0;
  uint32_t put = 0;
  rsx::Rsx::get().ea_to_io(k.r32(ctx + 8), &put);
  k.w32(ctrl, put);

  check(wait_until([&] { return k.r32(ctrl + 4) == put; }), "RSX consumed the FIFO up to put");
  rsx::Rsx::get().sync();  // GPU results -> guest memory before the test reads pixels
  check(k.r32(ctrl + 8) == 0x55, "SET_REFERENCE updated the ref register");
  check(k.r32(r.label_addr(5)) == 0x77, "semaphore release wrote label 5");
  check(ppu::ld32(k.base(), 0xC0000000 + fb_offset + 719 * pitch + 1279 * 4) == 0xFF336699, "clear filled the surface");
  check(wait_until([&] { return capture->frames == 1; }) && call("cellGcmSys", "cellGcmGetFlipStatus", {}).r[3] == 0,
        "flip presented one frame and set flip status to done");
  check(capture->width == 1280 && capture->height == 720 && capture->pixels[0] == 0xFF336699u,
        "presented pixels are the cleared colour (host BGRA)");
  check(wait_until([&] { return g_flip_handler_calls == 1; }), "guest flip handler called with head 1");

  // Segment wrap: the context callback moves to the next segment through a jump.
  const uint32_t before = k.r32(ctx + 8);
  call("cellGcmSys", "cellGcmCallback", {ctx, 0});
  const uint32_t seg2 = k.r32(ctx);
  check(seg2 == io_area + 0x1000 + 0x8000 && (ppu::ld32(k.base(), before) & 0xE0000003) == 0x20000000,
        "callback writes a jump and moves to the next segment");
  cur = seg2;
  emit(0x0050, {0x66});
  k.w32(ctx + 8, cur);
  rsx::Rsx::get().ea_to_io(cur, &put);
  k.w32(ctrl, put);
  check(wait_until([&] { return k.r32(ctrl + 8) == 0x66; }), "RSX followed the jump into the new segment");

  call("cellGcmSys", "cellGcmGetTiledPitchSize", {5000});
  check(call("cellGcmSys", "cellGcmGetTiledPitchSize", {5000}).r[3] == 0x1400, "tiled pitch rounds up");
  call("cellSysutil", "cellVideoOutGetResolution", {2, scratch + 0x300});
  check(ppu::ld16(k.base(), scratch + 0x300) == 1280 && ppu::ld16(k.base(), scratch + 0x302) == 720, "720p resolution");
  check(wait_until([&] { return r.stats.vblanks.load() > 2; }), "vblank timer running");
}

// ---------------------------------------------------------------------------------
// 3D draws through the FIFO
// ---------------------------------------------------------------------------------

// Vertex program encoder (field layout of the RSX VP instruction words).
constexpr uint32_t kSwzId = (3u << 8) | (2u << 10) | (1u << 12) | (0u << 14);
uint32_t vp_src(uint32_t type, uint32_t tmp = 0) { return type | (tmp << 2) | kSwzId; }
struct VpOp {
  uint32_t vec_op = 0, sca_op = 0, src[3] = {0, 0, 0}, input = 0, constant = 0;
  bool vec_result = false;
  uint32_t dst_tmp = 0x3F, dst_out = 0x1F, sca_dst_tmp = 0x3F, vec_mask = 0xF, sca_mask = 0;
  bool end = false;
};
std::array<uint32_t, 4> vp_encode(const VpOp& o) {
  const uint32_t s0 = o.src[0], s1 = o.src[1], s2 = o.src[2];
  const uint32_t d0 = (o.dst_tmp << 15) | (uint32_t(o.vec_result) << 30);
  const uint32_t d1 = ((s0 >> 9) & 0xFF) | (o.input << 8) | (o.constant << 12) | (o.vec_op << 22) | (o.sca_op << 27);
  const uint32_t d2 = ((s2 >> 11) & 0x3F) | ((s1 & 0x1FFFF) << 6) | ((s0 & 0x1FF) << 23);
  const uint32_t vm = o.vec_mask, sm = o.sca_mask;
  const uint32_t d3 = uint32_t(o.end) | (o.dst_out << 2) | (o.sca_dst_tmp << 7) | (((vm >> 3) & 1) << 13) |
                      (((vm >> 2) & 1) << 14) | (((vm >> 1) & 1) << 15) | ((vm & 1) << 16) | (((sm >> 3) & 1) << 17) |
                      (((sm >> 2) & 1) << 18) | (((sm >> 1) & 1) << 19) | ((sm & 1) << 20) | ((s2 & 0x7FF) << 21);
  return {d0, d1, d2, d3};
}

// Fragment program encoder; words are stored with their 16-bit halves swapped.
constexpr uint32_t kFpSwzId = (0u << 9) | (1u << 11) | (2u << 13) | (3u << 15);
uint32_t fp_swap(uint32_t v) { return (v << 16) | (v >> 16); }
struct FpOp {
  uint32_t op = 0, dst = 0, mask = 0xF, input = 0, tex = 0;
  uint32_t src_type[3] = {0, 0, 0}, src_index[3] = {0, 0, 0};
  bool end = false;
  std::array<float, 4> constant{};
};
uint32_t fp_write(uint8_t* mem, uint32_t addr, const std::vector<FpOp>& prog) {
  for (const FpOp& o : prog) {
    const uint32_t w0 = uint32_t(o.end) | (o.dst << 1) | (o.mask << 9) | (o.input << 13) | (o.tex << 17) |
                        ((o.op & 0x3F) << 24);
    uint32_t s[3];
    bool has_const = false;
    for (int k = 0; k < 3; ++k) {
      s[k] = o.src_type[k] | (o.src_index[k] << 2) | kFpSwzId;
      has_const |= o.src_type[k] == 2;
    }
    s[0] |= (7u << 18) | (0u << 21) | (1u << 23) | (2u << 25) | (3u << 27);  // cond = TR, swz = xyzw
    s[1] |= ((o.op >> 6) & 1) << 31;
    for (uint32_t w : {w0, s[0], s[1], s[2]}) { ppu::st32(mem, addr, fp_swap(w)); addr += 4; }
    if (has_const)
      for (float f : o.constant) { ppu::st32(mem, addr, fp_swap(std::bit_cast<uint32_t>(f))); addr += 4; }
  }
  return addr;
}

// Writes methods into the game command buffer and waits for the RSX to run them.
struct Fifo {
  lv2::Kernel& k = lv2::Kernel::get();
  std::vector<uint32_t> words;
  void method(uint32_t index, std::initializer_list<uint32_t> args) {
    words.push_back((uint32_t(args.size()) << 18) | (index * 4));
    words.insert(words.end(), args.begin(), args.end());
  }
  void f(uint32_t index, std::initializer_list<float> args) {
    words.push_back((uint32_t(args.size()) << 18) | (index * 4));
    for (float a : args) words.push_back(std::bit_cast<uint32_t>(a));
  }
  bool kick() {
    const uint32_t ctrl = rsx::Rsx::get().control_addr();
    size_t i = 0;
    while (i < words.size()) {
      if (k.r32(g_ctx + 8) + 64 > k.r32(g_ctx + 4)) call("cellGcmSys", "cellGcmCallback", {g_ctx, 0});
      uint32_t cur = k.r32(g_ctx + 8);
      ppu::st32(k.base(), cur, words[i++]);
      k.w32(g_ctx + 8, cur + 4);
    }
    words.clear();
    uint32_t put = 0;
    rsx::Rsx::get().ea_to_io(k.r32(g_ctx + 8), &put);
    k.w32(ctrl, put);
    const bool ok = wait_until([&] { return k.r32(ctrl + 4) == put; });
    rsx::Rsx::get().sync();  // GPU results -> guest memory before the test reads pixels
    return ok;
  }
};

// Framebuffer snapshots: the CPU pass records them, the GPU pass compares against them.
std::vector<std::vector<uint32_t>> g_reference;
size_t g_snapshot = 0;
bool g_gpu_pass = false;
int g_image_mismatches = 0;
uint32_t pixel(uint32_t x, uint32_t y);
void snapshot(const char* scenario) {
  std::vector<uint32_t> img(64 * 64);
  for (uint32_t y = 0; y < 64; ++y)
    for (uint32_t x = 0; x < 64; ++x) img[y * 64 + x] = pixel(x, y);
  if (!g_gpu_pass) {
    g_reference.push_back(img);
    return;
  }
  const auto& ref = g_reference[g_snapshot++];
  int bad = 0, worst = 0;
  for (size_t i = 0; i < img.size(); ++i)
    for (int c = 0; c < 32; c += 8) {
      const int dlt = std::abs(int((img[i] >> c) & 0xFF) - int((ref[i] >> c) & 0xFF));
      worst = std::max(worst, dlt);
      if (dlt > 2) { ++bad; break; }
    }
  char buf[160];
  std::snprintf(buf, sizeof buf, "GPU matches CPU reference: %s (%d px differ, max channel delta %d)", scenario, bad, worst);
  check(bad == 0, buf);
  g_image_mismatches += bad;
}

constexpr uint32_t kColor = 0x200000, kDepth = 0x300000, kFp = 0x400000, kVb = 0x500000, kTex = 0x600000,
                   kIb = 0x700000, kSize = 64, kPitch = kSize * 4;
uint32_t pixel(uint32_t x, uint32_t y) { return ppu::ld32(lv2::Kernel::get().base(), 0xC0000000 + kColor + y * kPitch + x * 4); }

void write_floats(uint32_t local_off, std::initializer_list<float> v) {
  uint32_t a = 0xC0000000 + local_off;
  for (float f : v) { ppu::st32(lv2::Kernel::get().base(), a, std::bit_cast<uint32_t>(f)); a += 4; }
}

void setup_target(Fifo& q) {
  using namespace rsx::m;
  q.method(CONTEXT_DMA_COLOR_A, {0xFEED0000});
  q.method(CONTEXT_DMA_ZETA, {0xFEED0000});
  q.method(SURFACE_CLIP_H, {kSize << 16, kSize << 16});
  q.method(SURFACE_FORMAT, {8 | (2 << 5) | (1 << 8), kPitch, kColor, kDepth});
  q.method(SURFACE_PITCH_Z, {kPitch});
  q.method(SURFACE_COLOR_TARGET, {1});
  q.f(VIEWPORT_OFFSET, {32.0f, 32.0f, 0.5f, 0.0f});
  q.f(VIEWPORT_SCALE, {32.0f, -32.0f, 0.5f, 0.0f});
  q.method(COLOR_CLEAR_VALUE, {0xFF000000});
  q.method(ZSTENCIL_CLEAR_VALUE, {0xFFFFFF00});
  q.method(CLEAR_SURFACE, {0xF3});
  q.method(BLEND_ENABLE, {0});
  q.method(DEPTH_TEST_ENABLE, {0});
}

// VP: o[HPOS] = v0 (optionally scaled by c0), o[COL0] = v3, o[TEX0] = v8
void load_vp(Fifo& q, bool scale_by_c0) {
  using namespace rsx::m;
  VpOp pos;
  pos.vec_op = scale_by_c0 ? 2 /*MUL*/ : 1 /*MOV*/;
  pos.src[0] = vp_src(2);
  pos.src[1] = vp_src(3);
  pos.input = 0;
  pos.constant = 0;
  pos.vec_result = true;
  pos.dst_out = 0;
  VpOp col;
  col.vec_op = 1;
  col.src[0] = vp_src(2);
  col.input = 3;
  col.vec_result = true;
  col.dst_out = 1;
  VpOp tex;
  tex.vec_op = 1;
  tex.src[0] = vp_src(2);
  tex.input = 8;
  tex.vec_result = true;
  tex.dst_out = 7;
  tex.end = true;
  q.method(TRANSFORM_PROGRAM_LOAD, {0});
  for (const VpOp& o : {pos, col, tex}) {
    const auto w = vp_encode(o);
    q.method(TRANSFORM_PROGRAM, {w[0], w[1], w[2], w[3]});
  }
  q.method(TRANSFORM_PROGRAM_START, {0});
}

void load_fp(Fifo& q, const std::vector<FpOp>& prog) {
  fp_write(lv2::Kernel::get().base(), 0xC0000000 + kFp, prog);
  q.method(rsx::m::SHADER_PROGRAM, {kFp | 1});
}

// Interleaved vertices: position (3 floats), colour (4 floats), uv (2 floats) = 36 bytes.
void bind_vertices(Fifo& q, uint32_t vb_off) {
  using namespace rsx::m;
  const uint32_t stride = 36;
  q.method(VERTEX_ARRAY_FORMAT + 0, {2 | (3 << 4) | (stride << 8)});
  q.method(VERTEX_ARRAY_FORMAT + 3, {2 | (4 << 4) | (stride << 8)});
  q.method(VERTEX_ARRAY_FORMAT + 8, {2 | (2 << 4) | (stride << 8)});
  q.method(VERTEX_ARRAY_OFFSET + 0, {vb_off});
  q.method(VERTEX_ARRAY_OFFSET + 3, {vb_off + 12});
  q.method(VERTEX_ARRAY_OFFSET + 8, {vb_off + 28});
}

void quad(uint32_t off, float z, std::array<float, 4> c, float scale = 1.0f) {
  const float s = scale;
  write_floats(off, {-s, s, z, c[0], c[1], c[2], c[3], 0, 0,
                     s, s, z, c[0], c[1], c[2], c[3], 1, 0,
                     s, -s, z, c[0], c[1], c[2], c[3], 1, 1,
                     -s, -s, z, c[0], c[1], c[2], c[3], 0, 1});
}

void draw_arrays(Fifo& q, uint32_t prim, uint32_t first, uint32_t count) {
  using namespace rsx::m;
  q.method(BEGIN_END, {prim});
  q.method(DRAW_ARRAYS, {first | ((count - 1) << 24)});
  q.method(BEGIN_END, {0});
}

const FpOp kFpColor = [] { FpOp o; o.op = 1; o.input = 1; o.src_type[0] = 1; o.end = true; return o; }();

void test_draws(bool gpu) {
  using namespace rsx::m;
  rsx::Rsx& r = rsx::Rsx::get();
  r.use_gpu = gpu;
  g_gpu_pass = gpu;
  g_snapshot = 0;
  if (gpu) {
    const std::string name = r.gpu_name();
    if (name.empty()) {
      std::printf("RSX 3D draws (GPU): no Vulkan device, skipped\n");
      return;
    }
    std::printf("RSX 3D draws (GPU renderer on %s)\n", name.c_str());
  } else {
    std::printf("RSX 3D draws (CPU reference renderer)\n");
  }
  const uint64_t gpu_draws_before = r.gpu_draws.load(), cpu_draws_before = r.cpu_draws.load();
  Fifo q;

  // 1. Flat-coloured triangle covering the lower-left half.
  setup_target(q);
  load_vp(q, false);
  load_fp(q, {kFpColor});
  write_floats(kVb, {-1, -1, 0, 0, 1, 0, 1, 0, 0,
                     1, -1, 0, 0, 1, 0, 1, 0, 0,
                     -1, 1, 0, 0, 1, 0, 1, 0, 0});
  bind_vertices(q, kVb);
  draw_arrays(q, 5, 0, 3);
  check(q.kick(), "FIFO with a draw executed");
  check(pixel(5, 40) == 0xFF00FF00 && pixel(60, 5) == 0xFF000000, "triangle: inside green, outside clear colour");
  check(r.stats.triangles.load() >= 1 && r.stats.skipped_draws.load() == 0, "triangle counted, no skipped draws");
  snapshot("triangle");

  // 2. Depth test: the near quad wins whichever order it is drawn in.
  setup_target(q);
  q.method(DEPTH_TEST_ENABLE, {1});
  q.method(DEPTH_FUNC, {0x201});
  quad(kVb, 0.6f, {0, 0, 1, 1});          // blue, far
  quad(kVb + 0x200, -0.2f, {1, 0, 0, 1}); // red, near
  bind_vertices(q, kVb);
  draw_arrays(q, 8, 0, 4);
  bind_vertices(q, kVb + 0x200);
  draw_arrays(q, 8, 0, 4);
  q.kick();
  const bool near_after = pixel(32, 32) == 0xFFFF0000;
  setup_target(q);
  q.method(DEPTH_TEST_ENABLE, {1});
  bind_vertices(q, kVb + 0x200);
  draw_arrays(q, 8, 0, 4);
  bind_vertices(q, kVb);
  draw_arrays(q, 8, 0, 4);
  q.kick();
  check(near_after && pixel(32, 32) == 0xFFFF0000, "depth test LESS keeps the nearer quad in both draw orders");
  snapshot("depth");

  // 3. Alpha blending: 50% white over black -> mid grey.
  setup_target(q);
  q.method(BLEND_ENABLE, {1});
  q.method(BLEND_SFACTOR, {0x0302 | (0x0302u << 16)});
  q.method(BLEND_DFACTOR, {0x0303 | (0x0303u << 16)});
  quad(kVb, 0.0f, {1, 1, 1, 0.5f});
  bind_vertices(q, kVb);
  draw_arrays(q, 8, 0, 4);
  q.kick();
  const uint32_t g = pixel(32, 32) & 0x00FFFFFF;
  if (g != 0x808080 && g != 0x7F7F7F) std::printf("  (blend result pixel = 0x%08X)\n", pixel(32, 32));
  check(g == 0x808080 || g == 0x7F7F7F, "blend SRC_ALPHA / ONE_MINUS_SRC_ALPHA");
  snapshot("blend");

  // 4. Texturing: 2x2 linear A8R8G8B8 texture, nearest filtering.
  setup_target(q);
  {
    uint8_t* m = lv2::Kernel::get().base();
    const uint32_t t = 0xC0000000 + kTex;
    ppu::st32(m, t + 0, 0xFFFF0000);   // red
    ppu::st32(m, t + 4, 0xFF00FF00);   // green
    ppu::st32(m, t + 8, 0xFF0000FF);   // blue
    ppu::st32(m, t + 12, 0xFFFFFFFF);  // white
  }
  q.method(TEXTURE_BASE + 0, {kTex});                              // offset
  q.method(TEXTURE_BASE + 1, {1 | (2 << 4) | ((0x85 | 0x20) << 8)});  // local, 2D, A8R8G8B8 linear
  q.method(TEXTURE_BASE + 2, {3 | (3 << 8)});                      // clamp
  q.method(TEXTURE_BASE + 3, {1u << 31});                          // enable
  q.method(TEXTURE_BASE + 5, {1u << 24});                          // nearest
  q.method(TEXTURE_BASE + 6, {(2u << 16) | 2});                    // 2 x 2
  FpOp texop;
  texop.op = 23;  // TEX
  texop.input = 4;  // TEX0
  texop.src_type[0] = 1;
  texop.tex = 0;
  texop.end = true;
  load_fp(q, {texop});
  quad(kVb, 0.0f, {1, 1, 1, 1});
  bind_vertices(q, kVb);
  draw_arrays(q, 8, 0, 4);
  q.kick();
  check(pixel(10, 10) == 0xFFFF0000 && pixel(50, 10) == 0xFF00FF00 && pixel(10, 50) == 0xFF0000FF &&
            pixel(50, 50) == 0xFFFFFFFF,
        "textured quad samples all four texels");
  snapshot("texture");
  q.method(TEXTURE_BASE + 3, {0});

  // 5. Indexed draw (u16) with a vertex program constant halving the quad.
  setup_target(q);
  load_vp(q, true);
  q.method(TRANSFORM_CONSTANT_LOAD, {0});
  q.f(TRANSFORM_CONSTANT, {0.5f, 0.5f, 1.0f, 1.0f});
  load_fp(q, {kFpColor});
  quad(kVb, 0.0f, {1, 1, 0, 1});
  {
    uint8_t* m = lv2::Kernel::get().base();
    const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
    for (int i = 0; i < 6; ++i) ppu::st16(m, 0xC0000000 + kIb + 2 * i, idx[i]);
  }
  bind_vertices(q, kVb);
  q.method(INDEX_ARRAY_ADDRESS, {kIb, 0 | (1 << 4)});  // address, then dma: local, u16
  q.method(BEGIN_END, {5});
  q.method(DRAW_INDEX_ARRAY, {0 | (5u << 24)});
  q.method(BEGIN_END, {0});
  q.kick();
  check(pixel(32, 32) == 0xFFFFFF00 && pixel(4, 4) == 0xFF000000 && pixel(20, 20) == 0xFFFFFF00,
        "indexed draw + MUL by transform constant shrinks the quad to the centre");
  snapshot("indexed + constants");
  load_vp(q, false);

  // 6. Fragment program with an inline constant (magenta).
  setup_target(q);
  FpOp konst;
  konst.op = 1;
  konst.src_type[0] = 2;
  konst.constant = {1, 0, 1, 1};
  konst.end = true;
  load_fp(q, {konst});
  quad(kVb, 0.0f, {0, 0, 0, 1});
  bind_vertices(q, kVb);
  draw_arrays(q, 8, 0, 4);
  q.kick();
  check(pixel(32, 32) == 0xFFFF00FF, "fragment program inline constant");
  snapshot("inline constant");

  // 7. A flip shows the rendered image.
  call("cellGcmSys", "cellGcmSetDisplayBuffer", {1, kColor, kPitch, kSize, kSize});
  call("cellGcmSys", "cellGcmSetFlipCommand", {g_ctx, 1});
  q.kick();
  auto cap = std::static_pointer_cast<gfx::CapturePresenter>(g_capture);
  check(wait_until([&] { return cap->width == kSize && cap->pixels[32 * kSize + 32] == 0xFFFF00FF; }),
        "flip presents the 3D-rendered frame");

  // 8. MSAA 2x1 (DIAGONAL_CENTERED_2): 32x64 logical clip/scissor/viewport scales to 64x64 samples.
  setup_target(q);
  q.method(SURFACE_CLIP_H, {(kSize / 2) << 16, kSize << 16});
  q.method(SURFACE_FORMAT, {8 | (2 << 5) | (1 << 8) | (3 << 12), kPitch, kColor, kDepth});
  q.method(SCISSOR_H, {(kSize / 2) << 16, kSize << 16});
  q.f(VIEWPORT_OFFSET, {kSize / 4.0f, kSize / 2.0f, 0.5f, 0.0f});
  q.f(VIEWPORT_SCALE, {kSize / 4.0f, -float(kSize) / 2.0f, 0.5f, 0.0f});
  q.method(CLEAR_SURFACE, {0xF3});
  load_fp(q, {kFpColor});
  quad(kVb, 0.0f, {0, 1, 1, 1});
  bind_vertices(q, kVb);
  draw_arrays(q, 8, 0, 4);
  q.method(SCISSOR_H, {0, 0});
  q.kick();
  check(pixel(8, 32) == 0xFF00FFFF && pixel(56, 32) == 0xFF00FFFF,
        "MSAA 2x1 scales clip, scissor and viewport across full buffer width");
  snapshot("msaa2x1");

  // 7. NV3062 + NV3089 2D/scaled image transfers (raw SRCCOPY + R5G6B5->A8R8G8B8 2x scale + GPU invalidation).
  auto emit_nv3089 = [&](uint32_t dst_off, uint32_t dst_fmt, uint32_t dst_pitch, uint32_t in_w, uint32_t in_h,
                         uint32_t out_w, uint32_t out_h, uint32_t ds_dx, uint32_t dt_dy, uint32_t src_off,
                         uint32_t src_fmt, uint32_t src_pitch, uint32_t op) {
    constexpr uint32_t kS3 = 0x6000 / 4, kS6 = 0xC000 / 4;
    q.method(kS3 + 0x188 / 4, {0xFEED0000});
    q.method(kS3 + 0x300 / 4, {dst_fmt});
    q.method(kS3 + 0x304 / 4, {(dst_pitch << 16) | dst_pitch});
    q.method(kS3 + 0x30C / 4, {dst_off});
    q.method(kS6 + 0x184 / 4, {0xFEED0000});
    q.method(kS6 + 0x198 / 4, {0x313371C3});
    q.method(kS6 + 0x300 / 4, {src_fmt});
    q.method(kS6 + 0x304 / 4, {op});
    q.method(kS6 + 0x308 / 4, {0});
    q.method(kS6 + 0x30C / 4, {(out_h << 16) | out_w});
    q.method(kS6 + 0x310 / 4, {0});
    q.method(kS6 + 0x314 / 4, {(out_h << 16) | out_w});
    q.method(kS6 + 0x318 / 4, {ds_dx});
    q.method(kS6 + 0x31C / 4, {dt_dy});
    q.method(kS6 + 0x400 / 4, {(in_h << 16) | in_w});
    q.method(kS6 + 0x404 / 4, {src_pitch});
    q.method(kS6 + 0x408 / 4, {src_off});
    q.method(kS6 + 0x40C / 4, {0});
  };
  {
    uint8_t* m = lv2::Kernel::get().base();
    constexpr uint32_t kSrcBuf = 0x800000, kDstBuf = 0x900000, kSrcImg = 0x810000, kDstImg = 0x910000;
    const uint32_t raw_words[8] = {0x3F800000u, 0xBF000000u, 0x00010002u, 0x00030004u,
                                   0xCAFEBABEu, 0x12345678u, 0x9ABCDEF0u, 0xDEAD0042u};
    for (uint32_t i = 0; i < 8; ++i) {
      ppu::st32(m, 0xC0000000 + kSrcBuf + i * 4, raw_words[i]);
      ppu::st32(m, 0xC0000000 + kDstBuf + i * 4, 0);
    }
    emit_nv3089(kDstBuf, 0x0A, 16, 4, 2, 4, 2, 0x00100000u, 0x00100000u, kSrcBuf, 3, 16, 3);
    // 2x2 R5G6B5 -> 4x4 A8R8G8B8 at 2x magnification (ds_dx = dt_dy = 0.5 in 12.20 fixed-point).
    ppu::st16(m, 0xC0000000 + kSrcImg + 0, 0xF800);
    ppu::st16(m, 0xC0000000 + kSrcImg + 2, 0x07E0);
    ppu::st16(m, 0xC0000000 + kSrcImg + 4, 0x001F);
    ppu::st16(m, 0xC0000000 + kSrcImg + 6, 0xFFFF);
    for (uint32_t i = 0; i < 16; ++i) ppu::st32(m, 0xC0000000 + kDstImg + i * 4, 0);
    emit_nv3089(kDstImg, 0x0A, 16, 2, 2, 4, 4, 0x00080000u, 0x00080000u, kSrcImg, 7, 4, 3);
    check(q.kick(), "NV3062+NV3089 blit commands executed");

    bool raw_ok = true;
    for (uint32_t i = 0; i < 8; ++i) raw_ok = raw_ok && ppu::ld32(m, 0xC0000000 + kDstBuf + i * 4) == raw_words[i];
    check(raw_ok, "NV3089 1:1 SRCCOPY preserves raw 32-bit vertex/index words");
    const uint32_t tl = ppu::ld32(m, 0xC0000000 + kDstImg + 0);
    const uint32_t tr = ppu::ld32(m, 0xC0000000 + kDstImg + 3 * 4);
    const uint32_t bl = ppu::ld32(m, 0xC0000000 + kDstImg + 3 * 16 + 0);
    const uint32_t br = ppu::ld32(m, 0xC0000000 + kDstImg + 3 * 16 + 3 * 4);
    check(tl == 0xFFFF0000 && tr == 0xFF00FF00 && bl == 0xFF0000FF && br == 0xFFFFFFFF,
          "NV3089 converts R5G6B5->A8R8G8B8 with 2x scaling");

    // Overwrite previously sampled kTex via NV3089 and draw to verify GPU cache invalidation.
    constexpr uint32_t kTexSrc = 0x820000;
    ppu::st32(m, 0xC0000000 + kTexSrc + 0, 0xFFFFFF00);
    ppu::st32(m, 0xC0000000 + kTexSrc + 4, 0xFFFF00FF);
    ppu::st32(m, 0xC0000000 + kTexSrc + 8, 0xFF00FFFF);
    ppu::st32(m, 0xC0000000 + kTexSrc + 12, 0xFF808080);
    setup_target(q);
    emit_nv3089(kTex, 0x0A, 8, 2, 2, 2, 2, 0x00100000u, 0x00100000u, kTexSrc, 3, 8, 3);
    q.method(TEXTURE_BASE + 0, {kTex});
    q.method(TEXTURE_BASE + 1, {1 | (2 << 4) | ((0x85 | 0x20) << 8)});
    q.method(TEXTURE_BASE + 2, {3 | (3 << 8)});
    q.method(TEXTURE_BASE + 3, {1u << 31});
    q.method(TEXTURE_BASE + 5, {1u << 24});
    q.method(TEXTURE_BASE + 6, {(2u << 16) | 2});
    load_fp(q, {texop});
    quad(kVb, 0.0f, {1, 1, 1, 1});
    bind_vertices(q, kVb);
    draw_arrays(q, 8, 0, 4);
    q.kick();
    check(pixel(10, 10) == 0xFFFFFF00 && pixel(50, 10) == 0xFFFF00FF && pixel(10, 50) == 0xFF00FFFF &&
              pixel(50, 50) == 0xFF808080,
          "NV3089 blit into VRAM invalidates cached GPU textures");
    snapshot("nv3089_tex");
    q.method(TEXTURE_BASE + 3, {0});
  }

  if (gpu) {
    check(r.gpu_draws.load() > gpu_draws_before && r.cpu_draws.load() == cpu_draws_before,
          "every draw in the GPU pass ran on the GPU (no CPU fallbacks)");
  } else {
    check(r.cpu_draws.load() > cpu_draws_before, "CPU pass used the reference renderer");
  }
  std::printf("  (%llu GPU draws, %llu CPU draws so far)\n", (unsigned long long)r.gpu_draws.load(),
              (unsigned long long)r.cpu_draws.load());
}

void test_audio() {
  using namespace lv2;
  Kernel& k = Kernel::get();
  std::printf("cellAudio\n");
  auto cap = std::make_unique<audio::CaptureOutput>();
  audio::CaptureOutput* captured = cap.get();
  audio::Mixer& mx = audio::Mixer::get();
  mx.start(std::move(cap), /*manual=*/true);

  const uint32_t g = k.heap.alloc(0x1000, 16);
  check(call("cellAudio", "cellAudioPortOpen", {g, g + 0x40}).r[3] == 0x80310703, "PortOpen before Init -> NOT_INIT");
  check(call("cellAudio", "cellAudioInit", {}).r[3] == 0, "Init");
  check(call("cellAudio", "cellAudioInit", {}).r[3] == 0x80310701, "second Init -> ALREADY_INIT");

  // CellAudioPortParam { u64 nChannel, u64 nBlock, u64 attr, float level }
  k.w64(g + 0, 2);
  k.w64(g + 8, 8);
  k.w64(g + 16, 0);
  check(call("cellAudio", "cellAudioPortOpen", {g, g + 0x40}).r[3] == 0, "open a 2ch, 8-block port");
  const uint32_t port = k.r32(g + 0x40);
  call("cellAudio", "cellAudioGetPortConfig", {port, g + 0x80});
  const uint32_t idx_addr = k.r32(g + 0x80), status = k.r32(g + 0x84), size = k.r32(g + 0x98), addr = k.r32(g + 0x9C);
  check(status == 1 && k.r64(g + 0x88) == 2 && k.r64(g + 0x90) == 8 && size == 2 * 8 * 256 * 4 && addr != 0,
        "port config: READY, 2 channels, 8 blocks, buffer size");

  // Block 0: constant L = 0.5, R = -0.25 (big-endian floats, as the game writes them).
  for (uint32_t f = 0; f < 256; ++f) {
    ppu::st32(k.base(), addr + 8 * f, std::bit_cast<uint32_t>(0.5f));
    ppu::st32(k.base(), addr + 8 * f + 4, std::bit_cast<uint32_t>(-0.25f));
  }
  const uint32_t ids = g + 0x100;
  call("cellAudio", "cellAudioCreateNotifyEventQueue", {ids, ids + 8});
  const uint32_t queue = k.r32(ids);
  call("cellAudio", "cellAudioSetNotifyEventQueue", {k.r64(ids + 8)});

  mx.step();  // port not started: silence, read index unchanged
  check(ppu::ld64(k.base(), idx_addr) == 0 && captured->samples.size() == 512 && captured->samples[0] == 0.0f,
        "stopped port is not mixed");
  check(call("cellAudio", "cellAudioPortStart", {port}).r[3] == 0, "PortStart");
  call("cellAudio", "cellAudioGetPortConfig", {port, g + 0x80});
  check(k.r32(g + 0x84) == 2, "status RUN after start");
  mx.step();
  check(captured->samples.size() == 1024 && captured->samples[512] == 0.5f && captured->samples[513] == -0.25f &&
            captured->samples[1022] == 0.5f,
        "mixer output carries the port's samples (L 0.5, R -0.25)");
  check(ppu::ld64(k.base(), idx_addr) == 1, "read index advanced to block 1");
  ppu::Context ev = sc(131, {queue, g + 0x200, 4, g + 0x300});
  check(ev.r[3] == 0 && k.r32(g + 0x300) == 2, "notify queue got one event per mixed period");

  // Port level (float argument in f1).
  {
    ppu::Context c;
    c.base = k.base();
    c.r[3] = port;
    c.f[1] = 0.5;
    hle::lookup("cellAudio", "cellAudioSetPortLevel")(c);
    check(c.r[3] == 0, "SetPortLevel");
  }
  for (uint32_t b = 0; b < 8; ++b)
    for (uint32_t f = 0; f < 256; ++f) {
      ppu::st32(k.base(), addr + b * 2048 + 8 * f, std::bit_cast<uint32_t>(0.8f));
      ppu::st32(k.base(), addr + b * 2048 + 8 * f + 4, std::bit_cast<uint32_t>(0.8f));
    }
  mx.step();
  check(std::fabs(captured->samples[1024] - 0.4f) < 1e-6f, "port level scales the output");
  check(call("cellAudio", "cellAudioPortStart", {port}).r[3] == 0x80310706, "start twice -> ALREADY_RUN");

  // 8-channel port: centre channel reaches both speakers at -3 dB.
  k.w64(g + 0, 8);
  k.w64(g + 8, 16);
  call("cellAudio", "cellAudioPortStop", {port});
  call("cellAudio", "cellAudioPortOpen", {g, g + 0x44});
  const uint32_t p8 = k.r32(g + 0x44);
  call("cellAudio", "cellAudioGetPortConfig", {p8, g + 0x80});
  const uint32_t a8 = k.r32(g + 0x9C);
  for (uint32_t f = 0; f < 256; ++f) ppu::st32(k.base(), a8 + 32 * f + 8, std::bit_cast<uint32_t>(1.0f));
  call("cellAudio", "cellAudioPortStart", {p8});
  captured->samples.clear();
  mx.step();
  check(p8 != port && std::fabs(captured->samples[0] - 0.70710678f) < 1e-5f &&
            std::fabs(captured->samples[1] - 0.70710678f) < 1e-5f,
        "8-channel downmix: centre -> L and R at 0.707");

  // Open ports until full (4 maximum).
  k.w64(g + 0, 2);
  k.w64(g + 8, 8);
  call("cellAudio", "cellAudioPortOpen", {g, g + 0x48});
  call("cellAudio", "cellAudioPortOpen", {g, g + 0x4C});
  check(call("cellAudio", "cellAudioPortOpen", {g, g + 0x50}).r[3] == 0x80310705, "fifth port -> PORT_FULL");
  check(call("cellAudio", "cellAudioQuit", {}).r[3] == 0, "Quit");
  mx.stop();
}

// ---- SPU thread groups ------------------------------------------------------
uint32_t sRR(uint32_t op, uint32_t rb, uint32_t ra, uint32_t rt) { return op << 21 | rb << 14 | ra << 7 | rt; }
uint32_t sRI7(uint32_t op, int32_t i7, uint32_t ra, uint32_t rt) { return op << 21 | (uint32_t(i7) & 0x7F) << 14 | ra << 7 | rt; }
uint32_t sRI10(uint32_t op, int32_t i10, uint32_t ra, uint32_t rt) { return op << 24 | (uint32_t(i10) & 0x3FF) << 14 | ra << 7 | rt; }
uint32_t sRI16(uint32_t op, int32_t i16, uint32_t rt) { return op << 23 | (uint32_t(i16) & 0xFFFF) << 7 | rt; }
uint32_t sRI18(uint32_t op, uint32_t i18, uint32_t rt) { return op << 25 | (i18 & 0x3FFFF) << 7 | rt; }
uint32_t wrch(uint32_t ch, uint32_t rt) { return sRR(0x10D, 0, ch, rt); }
uint32_t rdch(uint32_t ch, uint32_t rt) { return sRR(0x00D, 0, ch, rt); }
uint32_t il(int32_t v, uint32_t rt) { return sRI16(0x081, v, rt); }

void test_spu() {
  using namespace lv2;
  Kernel& k = Kernel::get();
  std::printf("spu thread groups\n");

  // SPU program: GET 16 bytes from arg1, add 1 to the first word, PUT it back;
  // send_event(port 5, 0x77, 0x88); read a mailbox value v; group_exit(v + 1).
  const std::vector<uint32_t> prog = {
      sRI18(0x21, 0x1000, 10),        // ila r10,0x1000
      wrch(16, 10),                   // MFC_LSA
      sRI7(0x1FC, 4, 3, 11),          // rotqbyi r11,r3,4  (arg1 low word -> preferred slot)
      wrch(18, 11),                   // MFC_EAL
      il(16, 12), wrch(19, 12),       // MFC_Size = 16
      il(0, 13), wrch(20, 13),        // MFC_TagID = 0
      il(0x40, 14), wrch(21, 14),     // MFC_Cmd = GET
      il(1, 15), wrch(22, 15), rdch(24, 16),  // tag mask, wait for tag status
      sRI10(0x34, 0, 10, 17),         // lqd r17,0(r10)
      sRI10(0x1C, 1, 17, 17),         // ai r17,r17,1
      sRI10(0x24, 0, 10, 17),         // stqd r17,0(r10)
      il(0x20, 14), wrch(21, 14),     // MFC_Cmd = PUT
      il(0x88, 20), wrch(28, 20),     // out mbox = data1
      sRI16(0x082, 0x0500, 21),       // ilhu r21,0x0500  (spup 5 << 24)
      sRI16(0x0C1, 0x0077, 21),       // iohl r21,0x77
      wrch(30, 21),                   // out interrupt mbox: send_event
      rdch(29, 22),                   // reply
      rdch(29, 18),                   // PPU mailbox value
      sRI10(0x1C, 1, 18, 18),         // ai r18,r18,1
      wrch(28, 18),                   // out mbox = status
      sRR(0x000, 0, 0, 0x101),        // stop 0x101 (group exit)
  };
  // Wrap it in a minimal 32-bit big-endian SPU ELF with one PT_LOAD at LS 0.
  std::vector<uint8_t> elf(0x100 + prog.size() * 4, 0);
  auto be32 = [&](size_t o, uint32_t v) { elf[o] = uint8_t(v >> 24); elf[o + 1] = uint8_t(v >> 16); elf[o + 2] = uint8_t(v >> 8); elf[o + 3] = uint8_t(v); };
  auto be16 = [&](size_t o, uint16_t v) { elf[o] = uint8_t(v >> 8); elf[o + 1] = uint8_t(v); };
  std::memcpy(elf.data(), "\x7F" "ELF\x01\x02\x01", 7);
  be16(16, 2); be16(18, 23);  // ET_EXEC, EM_SPU
  be32(24, 0);                // entry
  be32(28, 0x34);             // phoff
  be16(42, 32); be16(44, 1);  // phentsize, phnum
  be32(0x34, 1); be32(0x38, 0x100); be32(0x3C, 0); be32(0x40, 0);
  be32(0x44, uint32_t(prog.size() * 4)); be32(0x48, uint32_t(prog.size() * 4 + 0x40)); be32(0x4C, 7);
  for (size_t i = 0; i < prog.size(); ++i) be32(0x100 + i * 4, prog[i]);

  const uint32_t g = k.heap.alloc(0x2000, 128);
  std::memcpy(k.base() + g + 0x1000, elf.data(), elf.size());
  const uint32_t img = g, buf = g + 0x100, ids = g + 0x200, attr = g + 0x300, args = g + 0x400, ev = g + 0x500;
  k.w32(buf, 41);
  check(call("sysPrxForUser", "sys_spu_image_import", {img, g + 0x1000, 0}).r[3] == CELL_OK, "sys_spu_image_import");
  check(k.r32(img + 12) == 2, "image has a COPY and a FILL segment");

  k.w32(attr + 4, k.alloc_str("test_group"));
  check(sc(170, {ids, 1, 100, attr}).r[3] == CELL_OK, "sys_spu_thread_group_create");
  const uint32_t grp = k.r32(ids);
  k.w64(args, buf);
  check(sc(172, {ids + 4, grp, 0, img, 0, args}).r[3] == CELL_OK, "sys_spu_thread_initialize");
  const uint32_t th = k.r32(ids + 4);
  sc(128, {ids + 8, 0, 0x5350550000000001ull, 16});
  const uint32_t q = k.r32(ids + 8);
  check(sc(191, {th, q, 1, 5}).r[3] == CELL_OK, "sys_spu_thread_connect_event (port 5)");
  check(sc(173, {grp}).r[3] == CELL_OK, "sys_spu_thread_group_start");
  const ppu::Context e = sc(130, {q, ev, 0});
  check(e.r[3] == CELL_OK && e.r[4] == 0xFFFFFFFF53505501ull && e.r[5] == th && e.r[6] == (5ull << 32 | 0x77) && e.r[7] == 0x88,
        "SPU send_event arrives on the PPU event queue");
  sc(190, {th, 99});
  check(sc(178, {grp, ids + 12, ids + 16}).r[3] == CELL_OK, "sys_spu_thread_group_join");
  check(k.r32(ids + 12) == 1 && k.r32(ids + 16) == 100, "group exit cause GROUP_EXIT, status = mailbox + 1");
  check(k.r32(buf) == 42, "MFC GET/PUT round trip through main memory");
  check(sc(171, {grp}).r[3] == CELL_OK, "sys_spu_thread_group_destroy");
}

void test_spurs() {
  using namespace lv2;
  Kernel& k = Kernel::get();
  std::printf("spurs\n");
  // Policy module: increment the u32 at EA arg ($4), then exit to the kernel (bi $0).
  const std::vector<uint32_t> pm = {
      sRI18(0x21, 0x2000, 10), wrch(16, 10),       // MFC_LSA = 0x2000
      sRI7(0x1FC, 4, 4, 11), wrch(18, 11),         // MFC_EAL = arg
      il(16, 12), wrch(19, 12),                    // size 16
      il(0x40, 13), wrch(21, 13),                  // GET
      sRI10(0x34, 0, 10, 17), sRI10(0x1C, 1, 17, 17), sRI10(0x24, 0, 10, 17),
      il(0x20, 13), wrch(21, 13),                  // PUT
      sRR(0x1A8, 0, 0, 0),                         // bi $0  -> kernel exit
  };
  const uint32_t mem = k.heap.alloc(0x4000, 128);
  const uint32_t sp = mem, attr = mem + 0x1000, wattr = mem + 0x1200, code = mem + 0x1400, counter = mem + 0x1800,
                 widp = mem + 0x1880, prio = mem + 0x1890, frame = mem + 0x1900;
  for (size_t i = 0; i < pm.size(); ++i) k.w32(code + uint32_t(4 * i), pm[i]);
  k.w32(counter, 0);
  for (int i = 0; i < 8; ++i) k.base()[prio + i] = 1;

  check(call("cellSpurs", "_cellSpursAttributeInitialize", {attr, 2, 0x360001, 1, 100, 1000, 0}).r[3] == 0, "_cellSpursAttributeInitialize");
  check(call("cellSpurs", "cellSpursInitializeWithAttribute", {sp, attr}).r[3] == 0, "cellSpursInitializeWithAttribute (1 SPU)");
  // maxContention is the 9th argument: in the caller's parameter save area at r1+112.
  k.w64(frame + 112, 1);
  {
    ppu::Context c;
    c.base = k.base();
    const uint64_t a[] = {wattr, 1, 0x360001, code, uint32_t(pm.size() * 4), counter, prio, 1};
    for (int i = 0; i < 8; ++i) c.r[3 + i] = a[i];
    c.r[1] = frame;
    hle::lookup("cellSpurs", "_cellSpursWorkloadAttributeInitialize")(c);
    check(c.r[3] == 0 && k.r32(wattr + 0x24) == 1, "_cellSpursWorkloadAttributeInitialize (stack argument)");
  }
  check(call("cellSpurs", "cellSpursAddWorkloadWithAttribute", {sp, widp, wattr}).r[3] == 0 && k.r32(widp) == 0, "AddWorkloadWithAttribute -> wid 0");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(k.r32(counter) == 0, "workload with ready count 0 is not run");
  call("cellSpurs", "cellSpursReadyCountStore", {sp, 0, 1});
  for (int i = 0; i < 200 && k.r32(counter) < 10; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  check(k.r32(counter) >= 10, "ready workload's policy module runs repeatedly on the SPURS kernel");
  call("cellSpurs", "cellSpursReadyCountStore", {sp, 0, 0});
  check(call("cellSpurs", "cellSpursShutdownWorkload", {sp, 0}).r[3] == 0, "ShutdownWorkload");
  check(call("cellSpurs", "cellSpursWaitForWorkloadShutdown", {sp, 0}).r[3] == 0, "WaitForWorkloadShutdown");
  check(call("cellSpurs", "cellSpursRemoveWorkload", {sp, 0}).r[3] == 0, "RemoveWorkload");
  const uint32_t after = k.r32(counter);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(k.r32(counter) == after, "removed workload no longer runs");
  check(call("cellSpurs", "cellSpursFinalize", {sp}).r[3] == 0, "cellSpursFinalize");
}

// ---- cellSaveData ---------------------------------------------------------------
struct SaveTest {
  uint32_t data = 0, param = 0;
  int file_calls = 0;
  bool saving = false;
  uint32_t stat_isnew = 99, read_bytes = 0;
} g_save;

void save_stat_cb(ppu::Context& c) {  // (result, statGet, statSet)
  lv2::Kernel& k = lv2::Kernel::get();
  g_save.stat_isnew = k.r32(uint32_t(c.r[4]) + 4);
  if (g_save.saving) { k.w32(uint32_t(c.r[5]), g_save.param); k.w32(uint32_t(c.r[5]) + 4, 0); }
  k.w32(uint32_t(c.r[3]), 0);  // OK_NEXT
  g_save.file_calls = 0;
}
void save_file_cb(ppu::Context& c) {  // (result, fileGet, fileSet)
  lv2::Kernel& k = lv2::Kernel::get();
  const uint32_t fs = uint32_t(c.r[5]);
  if (g_save.file_calls++ == 0) {
    k.w32(fs, g_save.saving ? 1 : 0);  // WRITE / READ
    k.w32(fs + 8, 1);                  // NORMALFILE
    k.w32(fs + 28, k.alloc_str("SAVE.BIN"));
    k.w32(fs + 32, 0);
    k.w32(fs + 36, 16);
    k.w32(fs + 40, 16);
    k.w32(fs + 44, g_save.data);
    k.w32(uint32_t(c.r[3]), 0);  // OK_NEXT
  } else {
    g_save.read_bytes = k.r32(uint32_t(c.r[4]));  // excSize of the previous op
    k.w32(uint32_t(c.r[3]), 1);  // OK_LAST
  }
}

void test_savedata() {
  using namespace lv2;
  Kernel& k = Kernel::get();
  std::printf("savedata\n");
  const uint32_t m = k.heap.alloc(0x4000, 16);
  const uint32_t setBuf = m, buf = m + 0x100, data = m + 0x3000, param = m + 0x3100;
  k.w32(setBuf, 4); k.w32(setBuf + 4, 8); k.w32(setBuf + 32, 0x2000); k.w32(setBuf + 36, buf);
  std::memset(k.base() + param, 0, 1552);
  std::memcpy(k.base() + param, "Test Save", 9);
  g_save.data = data;
  g_save.param = param;
  const uint32_t stat = hle::make_opd(save_stat_cb), file = hle::make_opd(save_file_cb);
  const uint32_t dir = k.alloc_str("PS3RECOMP-TEST-SAVE");
  // Clean slate.
  if (auto r = vfs::resolve("/dev_hdd0/home/00000001/savedata/PS3RECOMP-TEST-SAVE")) std::filesystem::remove_all(r->host);

  for (int i = 0; i < 16; ++i) k.base()[data + i] = uint8_t(0xA0 + i);
  g_save.saving = true;
  check(call("cellSysutil", "cellSaveDataAutoSave2", {0, dir, 0, setBuf, stat, file, 0, 0}).r[3] == 0, "cellSaveDataAutoSave2");
  check(g_save.stat_isnew == 1, "first save reports isNewData");
  std::memset(k.base() + data, 0, 16);
  g_save.saving = false;
  check(call("cellSysutil", "cellSaveDataAutoLoad2", {0, dir, 0, setBuf, stat, file, 0, 0}).r[3] == 0, "cellSaveDataAutoLoad2");
  check(g_save.stat_isnew == 0, "existing save is not new");
  check(g_save.read_bytes == 16 && k.base()[data] == 0xA0 && k.base()[data + 15] == 0xAF, "file round-trips through the save directory");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: ps3_boot_test <test.elf>\n"); return 2; }
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep output if a check crashes
  test_nid();
  test_boot(argv[1]);
  test_kernel();
  test_fs_game_pad();
  test_gcm();
  test_draws(false);
  test_draws(true);
  rsx::Rsx::get().shutdown();
  test_audio();
  test_spu();
  test_spurs();
  test_savedata();
  if (g_fail) std::printf("%d of %d checks FAILED\n", g_fail, g_checks);
  else std::printf("all %d checks passed\n", g_checks);
  std::fflush(stdout);
  return g_fail ? 1 : 0;
}
