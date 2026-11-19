// cellFs (sys_fs) and cellGame HLE: guest file I/O through the virtual file system.
#define _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <optional>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <system_error>
#include <vector>

#include "hle.h"
#include "lv2.h"
#include "vfs.h"

namespace hle {
namespace {

namespace fs = std::filesystem;
using lv2::Kernel;
using lv2::CELL_OK;

// cellFs errors share the LV2 error space.
constexpr uint32_t CELL_FS_OK = 0;
constexpr uint32_t CELL_FS_EPERM = 0x80010009;
constexpr uint32_t CELL_FS_ENOENT = 0x80010006;
constexpr uint32_t CELL_FS_EINVAL = 0x80010002;
constexpr uint32_t CELL_FS_EEXIST = 0x80010014;
constexpr uint32_t CELL_FS_EIO = 0x8001002B;
constexpr uint32_t CELL_FS_EBADF = 0x8001002A;
constexpr uint32_t CELL_FS_EMFILE = 0x80010022;
constexpr uint32_t CELL_FS_EISDIR = 0x80010012;
constexpr uint32_t CELL_FS_ENOTDIR = 0x80010026;
constexpr uint32_t CELL_FS_ENOTEMPTY = 0x8001003C;

// open flags (octal, as in LV2)
constexpr uint32_t FS_O_ACCMODE = 3, FS_O_WRONLY = 1, FS_O_RDWR = 2;
constexpr uint32_t FS_O_CREAT = 0100, FS_O_EXCL = 0200, FS_O_TRUNC = 01000, FS_O_APPEND = 02000;

constexpr uint32_t FS_S_IFDIR = 0040000, FS_S_IFREG = 0100000;
constexpr uint32_t kDirentSize = 2 + 256;  // CellFsDirent: u8 type, u8 namlen, char name[256]

Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }

struct Handle {
  std::mutex m;
  bool tty = false;  // /dev_tty*: writes go to the guest console output
  std::FILE* fp = nullptr;
  fs::path path;
  bool append = false, writable = false;
  bool is_dir = false;
  std::vector<std::pair<std::string, uint8_t>> entries;  // name, type (1 dir, 2 file)
  size_t next = 0;
  ~Handle() { if (fp) std::fclose(fp); }
};

std::mutex g_m;
std::map<uint32_t, std::shared_ptr<Handle>> g_files;
uint32_t g_next_fd = 3;

uint32_t add_handle(std::shared_ptr<Handle> h) {
  std::lock_guard<std::mutex> lk(g_m);
  if (g_files.size() >= 1024) return 0;
  const uint32_t fd = g_next_fd++;
  g_files[fd] = std::move(h);
  return fd;
}
std::shared_ptr<Handle> get_handle(uint32_t fd) {
  std::lock_guard<std::mutex> lk(g_m);
  auto it = g_files.find(fd);
  return it == g_files.end() ? nullptr : it->second;
}
bool close_handle(uint32_t fd) {
  std::lock_guard<std::mutex> lk(g_m);
  return g_files.erase(fd) != 0;
}

std::FILE* open_host(const fs::path& p, const char* mode) {
#if defined(_WIN32)
  std::wstring wm(mode, mode + std::strlen(mode));
  return _wfopen(p.wstring().c_str(), wm.c_str());
#else
  return std::fopen(p.string().c_str(), mode);
#endif
}
int64_t tell64(std::FILE* f) {
#if defined(_WIN32)
  return _ftelli64(f);
#else
  return ftello(f);
#endif
}
int seek64(std::FILE* f, int64_t off, int whence) {
#if defined(_WIN32)
  return _fseeki64(f, off, whence);
#else
  return fseeko(f, off, whence);
#endif
}

// Writes CellFsStat (packed, 52 bytes):
//   s32 mode, s32 uid, s32 gid, s64 atime, s64 mtime, s64 ctime, u64 size, u64 blksize
void write_stat(uint32_t a, const fs::path& p) {
  std::error_code ec;
  const bool dir = fs::is_directory(p, ec);
  const uint64_t size = dir ? 0 : fs::file_size(p, ec);
  int64_t mtime = 0;
  const auto ft = fs::last_write_time(p, ec);
  if (!ec) {
    using namespace std::chrono;
    mtime = duration_cast<seconds>(clock_cast<system_clock>(ft).time_since_epoch()).count();
  }
  K().w32(a + 0, (dir ? FS_S_IFDIR : FS_S_IFREG) | 0777);
  K().w32(a + 4, 0);
  K().w32(a + 8, 0);
  K().w64(a + 12, uint64_t(mtime));
  K().w64(a + 20, uint64_t(mtime));
  K().w64(a + 28, uint64_t(mtime));
  K().w64(a + 36, size);
  K().w64(a + 44, 4096);
}

std::optional<vfs::Resolved> path_arg(ppu::Context& c, int n) { return vfs::resolve(K().read_str(a32(c, n), 1024)); }

// ---- files -------------------------------------------------------------------
void cellFsOpen_impl(ppu::Context& c);
void cellFsOpen(ppu::Context& c) {
  // r3 is overwritten with the result, so capture the arguments first.
  const std::string path = K().read_str(a32(c, 0), 256);
  const unsigned flags = a32(c, 1);
  cellFsOpen_impl(c);
  static const bool trace = std::getenv("PS3RECOMP_FS_TRACE") != nullptr;
  if (c.r[3] != CELL_FS_OK)
    K().logf("cellFsOpen(\"%s\", 0%o) failed: 0x%08X", path.c_str(), flags, unsigned(c.r[3]));
  else if (trace)
    K().logf("cellFsOpen(\"%s\", 0%o) ok", path.c_str(), flags);
}
void cellFsOpen_impl(ppu::Context& c) {
  // (const char* path, s32 flags, u32* fd, const void* arg, u64 size)
  if (K().read_str(a32(c, 0), 32).rfind("/dev_tty", 0) == 0) {  // console devices
    auto h = std::make_shared<Handle>();
    h->tty = true;
    h->writable = true;
    const uint32_t fd = add_handle(h);
    if (!fd) return ret(c, CELL_FS_EMFILE);
    K().w32(a32(c, 2), fd);
    return ret(c, CELL_FS_OK);
  }
  const auto r = path_arg(c, 0);
  if (!r) return ret(c, CELL_FS_ENOENT);
  const uint32_t flags = a32(c, 1);
  const uint32_t acc = flags & FS_O_ACCMODE;
  const bool write = acc == FS_O_WRONLY || acc == FS_O_RDWR;
  std::error_code ec;
  const bool exists = fs::exists(r->host, ec);

  if (write || (flags & (FS_O_CREAT | FS_O_TRUNC))) {
    if (r->read_only) return ret(c, CELL_FS_EPERM);
  }
  if (exists && fs::is_directory(r->host, ec)) return ret(c, CELL_FS_EISDIR);
  if (!exists && !(flags & FS_O_CREAT)) return ret(c, CELL_FS_ENOENT);
  // On a console an installed game's /dev_hdd0/game/<ID>/ folder already exists. Our
  // data folder starts empty, so create a game's own folder on its first write there.
  if (!exists && (flags & FS_O_CREAT) && !r->read_only && !fs::exists(r->host.parent_path(), ec) &&
      K().read_str(a32(c, 0), 64).rfind("/dev_hdd0/game/", 0) == 0) {
    fs::create_directories(r->host.parent_path(), ec);
  }
  if (exists && (flags & FS_O_CREAT) && (flags & FS_O_EXCL)) return ret(c, CELL_FS_EEXIST);

  const char* mode = "rb";
  if (write || (flags & FS_O_CREAT)) mode = (!exists || (flags & FS_O_TRUNC)) ? "w+b" : "r+b";
  else if (flags & FS_O_TRUNC) mode = "w+b";

  auto h = std::make_shared<Handle>();
  h->fp = open_host(r->host, mode);
  if (!h->fp) return ret(c, exists ? CELL_FS_EIO : CELL_FS_ENOENT);
  h->path = r->host;
  h->append = (flags & FS_O_APPEND) != 0;
  h->writable = write;
  const uint32_t fd = add_handle(h);
  if (!fd) return ret(c, CELL_FS_EMFILE);
  K().w32(a32(c, 2), fd);
  ret(c, CELL_FS_OK);
}

void cellFsClose(ppu::Context& c) { ret(c, close_handle(a32(c, 0)) ? CELL_FS_OK : CELL_FS_EBADF); }

void cellFsRead(ppu::Context& c) {
  // (u32 fd, void* buf, u64 nbytes, u64* nread)
  auto h = get_handle(a32(c, 0));
  if (!h || h->is_dir) return ret(c, CELL_FS_EBADF);
  if (h->tty) { K().w64(a32(c, 3), 0); return ret(c, CELL_FS_OK); }
  std::lock_guard<std::mutex> lk(h->m);
  const size_t n = h->fp ? std::fread(K().base() + a32(c, 1), 1, size_t(c.r[5]), h->fp) : 0;
  K().w64(a32(c, 3), n);
  ret(c, CELL_FS_OK);
}

void cellFsWrite(ppu::Context& c) {
  auto h = get_handle(a32(c, 0));
  if (!h || h->is_dir) return ret(c, CELL_FS_EBADF);
  if (h->tty) {
    const std::string s(reinterpret_cast<const char*>(K().base() + a32(c, 1)), size_t(c.r[5]));
    std::fflush(nullptr);
    if (K().tty) K().tty(s);
    else std::fwrite(s.data(), 1, s.size(), stdout);
    K().w64(a32(c, 3), c.r[5]);
    return ret(c, CELL_FS_OK);
  }
  if (!h->writable) return ret(c, CELL_FS_EBADF);
  std::lock_guard<std::mutex> lk(h->m);
  if (h->append) seek64(h->fp, 0, SEEK_END);
  const size_t n = std::fwrite(K().base() + a32(c, 1), 1, size_t(c.r[5]), h->fp);
  std::fflush(h->fp);
  K().w64(a32(c, 3), n);
  ret(c, CELL_FS_OK);
}

void cellFsLseek(ppu::Context& c) {
  // (u32 fd, s64 offset, s32 whence, u64* pos)
  auto h = get_handle(a32(c, 0));
  if (!h || h->is_dir) return ret(c, CELL_FS_EBADF);
  const int whence = int(int32_t(c.r[5]));
  if (whence < 0 || whence > 2) return ret(c, CELL_FS_EINVAL);
  std::lock_guard<std::mutex> lk(h->m);
  if (seek64(h->fp, int64_t(c.r[4]), whence == 0 ? SEEK_SET : whence == 1 ? SEEK_CUR : SEEK_END) != 0)
    return ret(c, CELL_FS_EINVAL);
  K().w64(a32(c, 3), uint64_t(tell64(h->fp)));
  ret(c, CELL_FS_OK);
}

void rw_with_offset(ppu::Context& c, bool write) {
  // (u32 fd, u64 offset, void* buf, u64 size, u64* n)
  auto h = get_handle(a32(c, 0));
  if (!h || h->is_dir || (write && !h->writable)) return ret(c, CELL_FS_EBADF);
  std::lock_guard<std::mutex> lk(h->m);
  const int64_t saved = tell64(h->fp);
  seek64(h->fp, int64_t(c.r[4]), SEEK_SET);
  uint8_t* buf = K().base() + a32(c, 2);
  const size_t n = write ? std::fwrite(buf, 1, size_t(c.r[6]), h->fp) : std::fread(buf, 1, size_t(c.r[6]), h->fp);
  if (write) std::fflush(h->fp);
  seek64(h->fp, saved, SEEK_SET);
  K().w64(a32(c, 4), n);
  ret(c, CELL_FS_OK);
}
void cellFsReadWithOffset(ppu::Context& c) { rw_with_offset(c, false); }
void cellFsWriteWithOffset(ppu::Context& c) { rw_with_offset(c, true); }

void cellFsFstat(ppu::Context& c) {
  auto h = get_handle(a32(c, 0));
  if (!h) return ret(c, CELL_FS_EBADF);
  if (h->fp) std::fflush(h->fp);
  write_stat(a32(c, 1), h->path);
  ret(c, CELL_FS_OK);
}
void cellFsStat(ppu::Context& c) {
  const auto r = path_arg(c, 0);
  std::error_code ec;
  if (!r || !fs::exists(r->host, ec)) return ret(c, CELL_FS_ENOENT);
  write_stat(a32(c, 1), r->host);
  ret(c, CELL_FS_OK);
}

void cellFsFtruncate(ppu::Context& c) {
  auto h = get_handle(a32(c, 0));
  if (!h || h->is_dir || !h->writable) return ret(c, CELL_FS_EBADF);
  std::fflush(h->fp);
  std::error_code ec;
  fs::resize_file(h->path, c.r[4], ec);
  ret(c, ec ? CELL_FS_EIO : CELL_FS_OK);
}
void cellFsTruncate(ppu::Context& c) {
  const auto r = path_arg(c, 0);
  if (!r) return ret(c, CELL_FS_ENOENT);
  if (r->read_only) return ret(c, CELL_FS_EPERM);
  std::error_code ec;
  if (!fs::exists(r->host, ec)) return ret(c, CELL_FS_ENOENT);
  fs::resize_file(r->host, c.r[4], ec);
  ret(c, ec ? CELL_FS_EIO : CELL_FS_OK);
}

// ---- directories ---------------------------------------------------------------
void cellFsOpendir(ppu::Context& c) {
  const auto r = path_arg(c, 0);
  std::error_code ec;
  if (!r || !fs::exists(r->host, ec)) return ret(c, CELL_FS_ENOENT);
  if (!fs::is_directory(r->host, ec)) return ret(c, CELL_FS_ENOTDIR);
  auto h = std::make_shared<Handle>();
  h->is_dir = true;
  h->path = r->host;
  h->entries = {{".", uint8_t(1)}, {"..", uint8_t(1)}};
  std::vector<std::pair<std::string, uint8_t>> items;
  for (const auto& e : fs::directory_iterator(r->host, ec))
    items.push_back({e.path().filename().string(), uint8_t(e.is_directory() ? 1 : 2)});
  std::sort(items.begin(), items.end());
  h->entries.insert(h->entries.end(), items.begin(), items.end());
  const uint32_t fd = add_handle(h);
  if (!fd) return ret(c, CELL_FS_EMFILE);
  K().w32(a32(c, 1), fd);
  ret(c, CELL_FS_OK);
}
void cellFsReaddir(ppu::Context& c) {
  // (u32 fd, CellFsDirent* dir, u64* nread)
  auto h = get_handle(a32(c, 0));
  if (!h || !h->is_dir) return ret(c, CELL_FS_EBADF);
  if (h->next >= h->entries.size()) {
    K().w64(a32(c, 2), 0);
    return ret(c, CELL_FS_OK);
  }
  const auto& [name, type] = h->entries[h->next++];
  uint8_t* d = K().base() + a32(c, 1);
  std::memset(d, 0, kDirentSize);
  const size_t n = std::min<size_t>(name.size(), 255);
  d[0] = type;
  d[1] = uint8_t(n);
  std::memcpy(d + 2, name.data(), n);
  K().w64(a32(c, 2), kDirentSize);
  ret(c, CELL_FS_OK);
}
void cellFsClosedir(ppu::Context& c) { ret(c, close_handle(a32(c, 0)) ? CELL_FS_OK : CELL_FS_EBADF); }

// cellFsGetDirectoryEntries(u32 fd, CellFsDirectoryEntry* entries, u32 entries_size, u32* data_count):
// readdir in batches, each entry = CellFsStat (52 bytes) + CellFsDirent (258), padded to 312.
void cellFsGetDirectoryEntries(ppu::Context& c) {
  constexpr uint32_t kEntrySize = 312;
  auto h = get_handle(a32(c, 0));
  if (!h || !h->is_dir) return ret(c, CELL_FS_EBADF);
  const uint32_t out = a32(c, 1), capacity = a32(c, 2) / kEntrySize;
  uint32_t n = 0;
  while (n < capacity && h->next < h->entries.size()) {
    const auto& [name, type] = h->entries[h->next++];
    const uint32_t e = out + n * kEntrySize;
    std::memset(K().base() + e, 0, kEntrySize);
    write_stat(e, name == "." || name == ".." ? h->path : h->path / fs::path(reinterpret_cast<const char8_t*>(name.c_str())));
    uint8_t* d = K().base() + e + 52;
    const size_t len = std::min<size_t>(name.size(), 255);
    d[0] = type;
    d[1] = uint8_t(len);
    std::memcpy(d + 2, name.data(), len);
    ++n;
  }
  K().w32(a32(c, 3), n);
  ret(c, CELL_FS_OK);
}

void cellFsMkdir(ppu::Context& c) {
  const auto r = path_arg(c, 0);
  if (!r) return ret(c, CELL_FS_ENOENT);
  if (r->read_only) return ret(c, CELL_FS_EPERM);
  std::error_code ec;
  if (fs::exists(r->host, ec)) return ret(c, CELL_FS_EEXIST);
  if (!fs::exists(r->host.parent_path(), ec)) return ret(c, CELL_FS_ENOENT);
  fs::create_directory(r->host, ec);
  ret(c, ec ? CELL_FS_EIO : CELL_FS_OK);
}
void cellFsRmdir(ppu::Context& c) {
  const auto r = path_arg(c, 0);
  std::error_code ec;
  if (!r || !fs::exists(r->host, ec)) return ret(c, CELL_FS_ENOENT);
  if (r->read_only) return ret(c, CELL_FS_EPERM);
  if (!fs::is_directory(r->host, ec)) return ret(c, CELL_FS_ENOTDIR);
  if (!fs::is_empty(r->host, ec)) return ret(c, CELL_FS_ENOTEMPTY);
  fs::remove(r->host, ec);
  ret(c, ec ? CELL_FS_EIO : CELL_FS_OK);
}
void cellFsUnlink(ppu::Context& c) {
  const auto r = path_arg(c, 0);
  std::error_code ec;
  if (!r || !fs::exists(r->host, ec)) return ret(c, CELL_FS_ENOENT);
  if (r->read_only) return ret(c, CELL_FS_EPERM);
  if (fs::is_directory(r->host, ec)) return ret(c, CELL_FS_EISDIR);
  fs::remove(r->host, ec);
  ret(c, ec ? CELL_FS_EIO : CELL_FS_OK);
}
void cellFsRename(ppu::Context& c) {
  const auto from = path_arg(c, 0), to = path_arg(c, 1);
  std::error_code ec;
  if (!from || !to || !fs::exists(from->host, ec)) return ret(c, CELL_FS_ENOENT);
  if (from->read_only || to->read_only) return ret(c, CELL_FS_EPERM);
  fs::rename(from->host, to->host, ec);
  ret(c, ec ? CELL_FS_EIO : CELL_FS_OK);
}

void cellFsGetFreeSize(ppu::Context& c) {
  // (const char* path, u32* block_size, u64* block_count)
  const auto r = path_arg(c, 0);
  if (!r) return ret(c, CELL_FS_ENOENT);
  std::error_code ec;
  const auto sp = fs::space(r->host, ec);
  K().w32(a32(c, 1), 4096);
  K().w64(a32(c, 2), ec ? 0 : sp.available / 4096);
  ret(c, CELL_FS_OK);
}
void cellFsGetBlockSize(ppu::Context& c) {
  // (const char* path, u64* sector_size, u64* block_size)
  K().w64(a32(c, 1), 512);
  K().w64(a32(c, 2), 4096);
  ret(c, CELL_FS_OK);
}
void fs_ok(ppu::Context& c) { ret(c, CELL_FS_OK); }

// ============================================================================
// cellGame
// ============================================================================
constexpr uint32_t CELL_GAME_GAMETYPE_DISC = 1;
constexpr uint32_t CELL_GAME_ERROR_PARAM = 0x8002CB07;

void write_cstr(uint32_t a, const std::string& s, size_t cap) {
  if (!a || !cap) return;
  const size_t n = std::min(s.size(), cap - 1);
  std::memcpy(K().base() + a, s.data(), n);
  K().base()[a + n] = 0;
}
// CellGameContentSize: s32 hddFreeSizeKB, s32 sizeKB, s32 sysSizeKB
void write_content_size(uint32_t a) {
  if (!a) return;
  K().w32(a + 0, 40 * 1024 * 1024);  // report 40 GB free
  K().w32(a + 4, uint32_t(-1));      // CELL_GAME_SIZEKB_NOTCALC
  K().w32(a + 8, 0);
}

void cellGameBootCheck(ppu::Context& c) {
  // (u32* type, u32* attributes, CellGameContentSize* size, char dirName[32])
  K().w32(a32(c, 0), CELL_GAME_GAMETYPE_DISC);
  K().w32(a32(c, 1), 0);
  write_content_size(a32(c, 2));
  write_cstr(a32(c, 3), "", 32);
  ret(c, CELL_OK);
}
void cellGameDataCheck(ppu::Context& c) {
  // (u32 type, const char* dirName, CellGameContentSize* size)
  write_content_size(a32(c, 2));
  ret(c, CELL_OK);
}
void cellGameContentPermit(ppu::Context& c) {
  // (char contentInfoPath[128], char usrdirPath[128])
  write_cstr(a32(c, 0), "/dev_bdvd/PS3_GAME", 128);
  write_cstr(a32(c, 1), "/dev_bdvd/PS3_GAME/USRDIR", 128);
  ret(c, CELL_OK);
}
void cellGameGetSizeKB(ppu::Context& c) { K().w32(a32(c, 0), 0); ret(c, CELL_OK); }

// Parameter ids -> PARAM.SFO keys (per the public cellGame documentation).
const char* string_param_key(uint32_t id, std::string& tmp) {
  switch (id) {
    case 0: return "TITLE";
    case 1: return "TITLE";  // TITLE_DEFAULT
    case 100: return "TITLE_ID";
    case 101: return "VERSION";
    case 105: return "PS3_SYSTEM_VER";
    case 106: return "APP_VER";
    default:
      if (id >= 2 && id <= 21) {  // localized titles TITLE_00 .. TITLE_19
        char b[16];
        std::snprintf(b, sizeof b, "TITLE_%02u", id - 2);
        tmp = b;
        return tmp.c_str();
      }
      return nullptr;
  }
}
void cellGameGetParamString(ppu::Context& c) {
  // (s32 id, char* buf, u32 bufsize)
  std::string tmp;
  const char* key = string_param_key(a32(c, 0), tmp);
  if (!key) return ret(c, CELL_GAME_ERROR_PARAM);
  const auto& s = vfs::game_sfo().strings;
  auto it = s.find(key);
  if (it == s.end() && a32(c, 0) >= 2 && a32(c, 0) <= 21) it = s.find("TITLE");  // fall back to default title
  write_cstr(a32(c, 1), it == s.end() ? std::string() : it->second, a32(c, 2));
  ret(c, CELL_OK);
}
void cellGameGetParamInt(ppu::Context& c) {
  // (s32 id, s32* value)
  const char* key = nullptr;
  switch (a32(c, 0)) {
    case 102: key = "PARENTAL_LEVEL"; break;
    case 103: key = "RESOLUTION"; break;
    case 104: key = "SOUND_FORMAT"; break;
    default: return ret(c, CELL_GAME_ERROR_PARAM);
  }
  const auto& m = vfs::game_sfo().ints;
  auto it = m.find(key);
  K().w32(a32(c, 1), it == m.end() ? 0 : it->second);
  ret(c, CELL_OK);
}

}  // namespace

void register_fs_modules() {
#define R(mod, fn) register_function(mod, #fn, fn)
  const char* F = "sys_fs";
  R(F, cellFsOpen); R(F, cellFsClose); R(F, cellFsRead); R(F, cellFsWrite); R(F, cellFsLseek);
  R(F, cellFsReadWithOffset); R(F, cellFsWriteWithOffset); R(F, cellFsFstat); R(F, cellFsStat);
  R(F, cellFsFtruncate); R(F, cellFsTruncate); R(F, cellFsOpendir); R(F, cellFsReaddir); R(F, cellFsClosedir); R(F, cellFsGetDirectoryEntries);
  R(F, cellFsMkdir); R(F, cellFsRmdir); R(F, cellFsUnlink); R(F, cellFsRename); R(F, cellFsGetFreeSize);
  R(F, cellFsGetBlockSize);
  register_function(F, "cellFsFGetBlockSize", [](ppu::Context& c) {
    K().w64(a32(c, 1), 512);
    K().w64(a32(c, 2), 4096);
    ret(c, CELL_FS_OK);
  });
  register_function(F, "cellFsChmod", fs_ok);
  register_function(F, "cellFsUtime", fs_ok);
  register_function(F, "cellFsSetIoBufferFromDefaultContainer", fs_ok);

  const char* G = "cellGame";
  R(G, cellGameBootCheck); R(G, cellGameDataCheck); R(G, cellGameContentPermit); R(G, cellGameGetSizeKB);
  R(G, cellGameGetParamString); R(G, cellGameGetParamInt);
#undef R
}

}  // namespace hle
