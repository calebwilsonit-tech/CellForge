// cellSaveData (exported by cellSysutil): callback-driven save/load.
//
// Each save is a directory /dev_hdd0/home/00000001/savedata/<dirName>/ in the
// writable data folder. The game's files are stored as plain host files; the
// system file parameters (title, subtitle, detail, list param) are kept in a
// PARAM.PS3RECOMP sidecar (this is our own format, not a console PARAM.SFO).
//
// Flow (as the system library drives it): pick a directory (auto: given; list:
// funcList; fixed: funcFixed) -> funcStat with the directory's status -> funcFile
// repeatedly, performing one file operation per call, until OK_LAST.
// Structure layouts follow the public SDK headers.

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "hle.h"
#include "lv2.h"
#include "vfs.h"

namespace hle {
namespace {

namespace fs = std::filesystem;
using lv2::Kernel;
Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
uint8_t* G(uint32_t a) { return K().base() + a; }

constexpr uint32_t kErrCbResult = 0x8002B401, kErrAccess = 0x8002B402, kErrParam = 0x8002B404;
constexpr uint32_t kRetCancel = 1;
constexpr int32_t kOkNext = 0, kOkLast = 1, kOkLastNoConfirm = 2;
constexpr uint32_t kOpRead = 0, kOpWrite = 1, kOpDelete = 2, kOpWriteNoTrunc = 3;
constexpr uint32_t kParamSize = 1552;  // CellSaveDataSystemFileParam
constexpr const char* kRoot = "/dev_hdd0/home/00000001/savedata";
constexpr const char* kParamFile = "PARAM.PS3RECOMP";

std::string cstr(uint32_t a, size_t max) { return a ? K().read_str(a, max) : std::string(); }
void put_str(uint32_t a, const std::string& s, size_t field) {
  std::memset(G(a), 0, field);
  std::memcpy(G(a), s.data(), std::min(s.size(), field - 1));
}

std::optional<fs::path> host_dir(const std::string& dir) {
  if (dir.empty() || dir.find('/') != std::string::npos || dir.find("..") != std::string::npos) return std::nullopt;
  auto r = vfs::resolve(std::string(kRoot) + "/" + dir);
  if (!r) return std::nullopt;
  return r->host;
}

int64_t to_time(fs::file_time_type t) {
  using namespace std::chrono;
  const auto sys = time_point_cast<seconds>(clock_cast<system_clock>(t));
  return sys.time_since_epoch().count();
}

// Guest scratch structures for one operation.
struct Scratch {
  uint32_t base = 0;
  uint32_t result, listGet, listSet, fixedSet, statGet, statSet, fileGet, fileSet;
  Scratch() {
    base = K().hle.alloc(0x1000, 16);
    std::memset(G(base), 0, 0x1000);
    result = base;              // 0x20
    listGet = base + 0x20;      // 0x50
    listSet = base + 0x70;      // 0x20
    fixedSet = base + 0x90;     // 0x10
    fileGet = base + 0xA0;      // 0x50
    fileSet = base + 0xF0;      // 0x30
    statSet = base + 0x120;     // 0x10
    statGet = base + 0x140;     // 1704
  }
  ~Scratch() { K().hle.free(base); }
};

int32_t call_cb(uint32_t fn, std::initializer_list<uint64_t> args, uint32_t result) {
  K().w32(result, 0);
  K().call_guest(fn, args);
  return int32_t(K().r32(result));
}

std::vector<std::string> list_dirs(const std::string& prefixes) {
  std::vector<std::string> out;
  auto root = vfs::resolve(kRoot);
  if (!root) return out;
  std::vector<std::string> pre;
  size_t p = 0;
  while (true) {  // "|" separates several prefixes
    const size_t q = prefixes.find('|', p);
    pre.push_back(prefixes.substr(p, q == std::string::npos ? std::string::npos : q - p));
    if (q == std::string::npos) break;
    p = q + 1;
  }
  std::error_code ec;
  for (auto& e : fs::directory_iterator(root->host, ec)) {
    if (!e.is_directory()) continue;
    const std::string n = e.path().filename().string();
    for (auto& x : pre)
      if (n.rfind(x, 0) == 0) { out.push_back(n); break; }
  }
  std::sort(out.begin(), out.end());
  return out;
}

void read_params(const fs::path& d, uint32_t dst) {
  std::memset(G(dst), 0, kParamSize);
  std::ifstream in(d / kParamFile, std::ios::binary);
  if (in) in.read(reinterpret_cast<char*>(G(dst)), kParamSize);
}

std::string file_name(uint32_t type, const std::string& name) {
  switch (type) {
    case 2: return "ICON0.PNG";
    case 3: return "ICON1.PAM";
    case 4: return "PIC1.PNG";
    case 5: return "SND0.AT3";
    default: return name;
  }
}

// Runs stat + file callbacks for `dir`. save: creating/writing is allowed.
uint32_t run_dir(const Scratch& s, const std::string& dir, bool save, uint32_t setBuf, uint32_t funcStat, uint32_t funcFile,
                 uint32_t userdata, uint32_t buf_off) {
  const auto hd = host_dir(dir);
  if (!hd) return kErrParam;
  std::error_code ec;
  const bool exists = fs::exists(*hd, ec);
  K().logf("cellSaveData: %s '%s' (%s), funcStat 0x%X funcFile 0x%X", save ? "save" : "load", dir.c_str(),
           exists ? "exists" : "new", K().r32(funcStat), K().r32(funcFile));

  // ---- stat ----
  const uint32_t sg = s.statGet;
  std::memset(G(sg), 0, 1704);
  K().w32(sg + 0, 10 * 1024 * 1024);  // hddFreeSizeKB: 10 GB
  K().w32(sg + 4, exists ? 0 : 1);     // isNewData
  put_str(sg + 8 + 24, dir, 32);
  if (exists) {
    const int64_t t = to_time(fs::last_write_time(*hd, ec));
    K().w64(sg + 8, uint64_t(t)); K().w64(sg + 16, uint64_t(t)); K().w64(sg + 24, uint64_t(t));
    read_params(*hd, sg + 64);
  }
  // File list goes into the game's buffer after the directory list area.
  const uint32_t file_max = K().r32(setBuf + 4), buf = K().r32(setBuf + 36), buf_size = K().r32(setBuf + 32);
  uint32_t nfiles = 0, nlisted = 0, sizeKB = 0;
  if (exists) {
    for (auto& e : fs::directory_iterator(*hd, ec)) {
      if (!e.is_regular_file() || e.path().filename() == kParamFile) continue;
      ++nfiles;
      const uint64_t sz = e.file_size(ec);
      sizeKB += uint32_t((sz + 1023) / 1024);
      const uint32_t fe = buf + buf_off + 56 * nlisted;
      if (nlisted < file_max && buf && fe + 56 <= buf + buf_size) {
        std::memset(G(fe), 0, 56);
        const std::string n = e.path().filename().string();
        uint32_t type = 1;  // normal file
        if (n == "ICON0.PNG") type = 2; else if (n == "ICON1.PAM") type = 3; else if (n == "PIC1.PNG") type = 4; else if (n == "SND0.AT3") type = 5;
        K().w32(fe, type);
        K().w64(fe + 8, sz);
        const int64_t t = to_time(e.last_write_time(ec));
        K().w64(fe + 16, uint64_t(t)); K().w64(fe + 24, uint64_t(t)); K().w64(fe + 32, uint64_t(t));
        put_str(fe + 40, n, 13);
        ++nlisted;
      }
    }
  }
  K().w32(sg + 1620, sizeKB);
  K().w32(sg + 1624, 35);  // sysSizeKB
  K().w32(sg + 1628, nfiles);
  K().w32(sg + 1632, nlisted);
  K().w32(sg + 1636, buf ? buf + buf_off : 0);
  std::memset(G(s.statSet), 0, 16);
  int32_t r = call_cb(funcStat, {s.result, sg, s.statSet}, s.result);
  if (r < 0) { K().logf("cellSaveData: stat callback returned %d", r); return kErrCbResult; }

  if (const uint32_t param = K().r32(s.statSet)) {  // setParam: (re)write the system file parameters
    fs::create_directories(*hd, ec);
    std::ofstream out(*hd / kParamFile, std::ios::binary);
    out.write(reinterpret_cast<const char*>(G(param)), kParamSize);
  }
  const uint32_t recreate = K().r32(s.statSet + 4) & 0xFFFEFFFF;
  K().logf("cellSaveData:   stat cb -> %d, setParam %s, reCreateMode 0x%X", r, K().r32(s.statSet) ? "yes" : "no",
           K().r32(s.statSet + 4));
  if (recreate >= 2 && fs::exists(*hd, ec)) {  // RECREATE_YES[_RESET_OWNER]: start from an empty directory
    for (auto& e : fs::directory_iterator(*hd, ec))
      if (e.path().filename() != kParamFile) fs::remove_all(e.path(), ec);
  }
  if (r == kOkLast || r == kOkLastNoConfirm) return 0;
  if (save) fs::create_directories(*hd, ec);

  // ---- files ----
  for (int guard = 0; guard < 100000; ++guard) {
    std::memset(G(s.fileSet), 0, 0x30);
    r = call_cb(funcFile, {s.result, s.fileGet, s.fileSet}, s.result);
    if (r < 0) { K().logf("cellSaveData: file callback returned %d", r); return kErrCbResult; }
    if (r == kOkLast || r == kOkLastNoConfirm) break;
    const uint32_t fsx = s.fileSet;
    const uint32_t op = K().r32(fsx), type = K().r32(fsx + 8);
    const std::string name = file_name(type, cstr(K().r32(fsx + 28), 13));
    const uint32_t off = K().r32(fsx + 32), size = K().r32(fsx + 36), bsize = K().r32(fsx + 40), fbuf = K().r32(fsx + 44);
    if (name.empty() || name.find('/') != std::string::npos || name.find("..") != std::string::npos) return kErrParam;
    const fs::path file = *hd / name;
    uint32_t done = 0;
    switch (op) {
      case kOpRead: {
        std::ifstream in(file, std::ios::binary);
        if (in) {
          in.seekg(off);
          in.read(reinterpret_cast<char*>(G(fbuf)), std::min(size, bsize));
          done = uint32_t(in.gcount());
        } else {
          K().logf("cellSaveData: read of missing file %s", name.c_str());
        }
        break;
      }
      case kOpWrite: case kOpWriteNoTrunc: {
        if (!save) return kErrParam;
        fs::create_directories(*hd, ec);
        std::fstream out;
        if (op == kOpWrite) out.open(file, std::ios::binary | std::ios::out | std::ios::trunc);
        else {
          out.open(file, std::ios::binary | std::ios::in | std::ios::out);
          if (!out) out.open(file, std::ios::binary | std::ios::out);
        }
        if (!out) return kErrAccess;
        out.seekp(off);
        out.write(reinterpret_cast<const char*>(G(fbuf)), std::min(size, bsize));
        done = std::min(size, bsize);
        break;
      }
      case kOpDelete:
        fs::remove(file, ec);
        break;
      default:
        return kErrParam;
    }
    K().w32(s.fileGet, done);  // excSize
    K().logf("cellSaveData:   file op %u %s off %u size %u -> %u", op, name.c_str(), off, std::min(size, bsize), done);
  }
  (void)userdata;
  return 0;
}

// ---- entry points ----
// AutoLoad2/AutoSave2(version, dirName, errDialog, setBuf, funcStat, funcFile, container, userdata)
uint32_t auto_op(ppu::Context& c, bool save) {
  Scratch s;
  K().logf("cellSaveDataAuto%s", save ? "Save2" : "Load2");
  K().w32(s.result + 16, a32(c, 7));  // userdata
  return run_dir(s, cstr(a32(c, 1), 32), save, a32(c, 3), a32(c, 4), a32(c, 5), a32(c, 7), 0);
}
void cellSaveDataAutoLoad2(ppu::Context& c) { c.r[3] = auto_op(c, false); }
void cellSaveDataAutoSave2(ppu::Context& c) { c.r[3] = auto_op(c, true); }

// Lists matching directories into setBuf->buf and fills a CellSaveDataListGet.
uint32_t fill_list(const Scratch& s, uint32_t setList, uint32_t setBuf, std::vector<std::string>& dirs) {
  dirs = list_dirs(cstr(K().r32(setList + 8), 256));
  const uint32_t dir_max = K().r32(setBuf), buf = K().r32(setBuf + 36), buf_size = K().r32(setBuf + 32);
  uint32_t n = 0;
  for (auto& d : dirs) {
    const uint32_t e = buf + 48 * n;
    if (n >= dir_max || !buf || e + 48 > buf + buf_size) break;
    std::memset(G(e), 0, 48);
    put_str(e, d, 32);
    if (auto hd = host_dir(d)) {
      std::ifstream in(*hd / kParamFile, std::ios::binary);
      if (in) { in.seekg(1288); in.read(reinterpret_cast<char*>(G(e + 32)), 8); }
    }
    ++n;
  }
  K().w32(s.listGet, uint32_t(dirs.size()));
  K().w32(s.listGet + 4, n);
  K().w32(s.listGet + 8, buf);
  return 48 * n;  // bytes used in the buffer
}

// ListLoad2/ListSave2(version, setList, setBuf, funcList, funcStat, funcFile, container, userdata)
uint32_t list_op(ppu::Context& c, bool save) {
  Scratch s;
  K().w32(s.result + 16, a32(c, 7));
  std::vector<std::string> dirs;
  const uint32_t used = fill_list(s, a32(c, 1), a32(c, 2), dirs);
  const int32_t r = call_cb(a32(c, 3), {s.result, s.listGet, s.listSet}, s.result);
  if (r < 0) { K().logf("cellSaveDataList%s: list callback returned %d", save ? "Save" : "Load", r); return kErrCbResult; }
  // No list UI: take the new-data entry when saving, else the focused or first
  // fixed entry.
  std::string dir;
  const uint32_t newData = K().r32(s.listSet + 16);
  if (save && newData) dir = cstr(K().r32(newData + 4), 32);
  if (dir.empty() && K().r32(s.listSet + 4)) dir = cstr(K().r32(s.listSet + 4), 32);
  if (dir.empty() && K().r32(s.listSet + 8)) dir = cstr(K().r32(s.listSet + 12), 32);
  K().logf("cellSaveDataList%s: prefix '%s', %zu dir(s), list cb -> %d, focus %u, fixed %u, newData %u -> '%s'",
           save ? "Save" : "Load", cstr(K().r32(a32(c, 1) + 8), 256).c_str(), dirs.size(), r, K().r32(s.listSet),
           K().r32(s.listSet + 8), newData, dir.c_str());
  if (dir.empty()) return kRetCancel;  // nothing to load: as if the user cancelled
  return run_dir(s, dir, save, a32(c, 2), a32(c, 4), a32(c, 5), a32(c, 7), (used + 15) & ~15u);
}
void cellSaveDataListLoad2(ppu::Context& c) { c.r[3] = list_op(c, false); }
void cellSaveDataListSave2(ppu::Context& c) { c.r[3] = list_op(c, true); }

// FixedLoad2/FixedSave2(version, setList, setBuf, funcFixed, funcStat, funcFile, container, userdata)
uint32_t fixed_op(ppu::Context& c, bool save) {
  Scratch s;
  K().w32(s.result + 16, a32(c, 7));
  std::vector<std::string> dirs;
  const uint32_t used = fill_list(s, a32(c, 1), a32(c, 2), dirs);
  const int32_t r = call_cb(a32(c, 3), {s.result, s.listGet, s.fixedSet}, s.result);
  if (r < 0) { K().logf("cellSaveDataFixed%s: fixed callback returned %d", save ? "Save" : "Load", r); return kErrCbResult; }
  const std::string dir = cstr(K().r32(s.fixedSet), 32);
  K().logf("cellSaveDataFixed%s: prefix '%s', %zu dir(s), fixed cb -> %d -> '%s'", save ? "Save" : "Load",
           cstr(K().r32(a32(c, 1) + 8), 256).c_str(), dirs.size(), r, dir.c_str());
  if (dir.empty()) return kErrParam;
  return run_dir(s, dir, save, a32(c, 2), a32(c, 4), a32(c, 5), a32(c, 7), (used + 15) & ~15u);
}
void cellSaveDataFixedLoad2(ppu::Context& c) { c.r[3] = fixed_op(c, false); }
void cellSaveDataFixedSave2(ppu::Context& c) { c.r[3] = fixed_op(c, true); }

}  // namespace

void register_savedata_module() {
#define R(fn) register_function("cellSysutil", #fn, fn)
  R(cellSaveDataAutoLoad2);
  R(cellSaveDataAutoSave2);
  R(cellSaveDataListLoad2);
  R(cellSaveDataListSave2);
  R(cellSaveDataFixedLoad2);
  R(cellSaveDataFixedSave2);
#undef R
}

}  // namespace hle
