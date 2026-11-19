#include "vfs.h"

#include <fstream>
#include <mutex>
#include <vector>

namespace vfs {
namespace {

std::mutex g_m;
std::map<std::string, Mount> g_mounts;
Sfo g_sfo;

uint32_t le32(const std::vector<uint8_t>& b, size_t o) {
  return uint32_t(b[o]) | (uint32_t(b[o + 1]) << 8) | (uint32_t(b[o + 2]) << 16) | (uint32_t(b[o + 3]) << 24);
}
uint16_t le16(const std::vector<uint8_t>& b, size_t o) { return uint16_t(b[o] | (b[o + 1] << 8)); }

}  // namespace

void mount(const std::string& device, const fs::path& host, bool read_only) {
  std::lock_guard<std::mutex> lk(g_m);
  g_mounts[device] = Mount{fs::weakly_canonical(host), read_only};
}

void unmount_all() {
  std::lock_guard<std::mutex> lk(g_m);
  g_mounts.clear();
}

std::optional<Resolved> resolve(std::string_view guest) {
  if (guest.empty() || guest[0] != '/') return std::nullopt;
  // Split into device ("/dev_hdd0") and the rest.
  const size_t slash = guest.find('/', 1);
  const std::string device(guest.substr(0, slash));
  const std::string rest = slash == std::string_view::npos ? std::string() : std::string(guest.substr(slash + 1));

  Mount m;
  {
    std::lock_guard<std::mutex> lk(g_m);
    auto it = g_mounts.find(device);
    if (it == g_mounts.end()) return std::nullopt;
    m = it->second;
  }
  // Normalise lexically, then make sure the result stays under the mount root.
  fs::path rel = fs::path(rest).lexically_normal();
  if (rel == ".") rel.clear();
  for (const auto& part : rel)
    if (part == "..") return std::nullopt;
  if (rel.has_root_path()) return std::nullopt;
  return Resolved{rel.empty() ? m.host : m.host / rel, m.read_only};
}

void mount_standard(const fs::path& game_root, const fs::path& data_root) {
  std::error_code ec;
  for (const char* d : {"dev_hdd0", "dev_hdd1", "dev_usb000"}) fs::create_directories(data_root / d, ec);
  fs::create_directories(data_root / "dev_hdd0" / "home" / "00000001" / "savedata", ec);
  fs::create_directories(data_root / "dev_hdd0" / "game", ec);
  mount("/dev_bdvd", game_root, true);
  mount("/app_home", game_root / "PS3_GAME" / "USRDIR", true);
  mount("/dev_hdd0", data_root / "dev_hdd0", false);
  mount("/dev_hdd1", data_root / "dev_hdd1", false);
  mount("/dev_usb000", data_root / "dev_usb000", false);

  Sfo sfo;
  if (sfo.load(game_root / "PS3_GAME" / "PARAM.SFO")) set_game_sfo(sfo);
}

bool Sfo::load(const fs::path& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) return false;
  std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (b.size() < 0x14 || le32(b, 0) != 0x46535000) return false;  // "\0PSF"
  const uint32_t key_table = le32(b, 8), data_table = le32(b, 12), count = le32(b, 16);
  if (size_t(0x14) + size_t(count) * 16 > b.size()) return false;
  for (uint32_t i = 0; i < count; ++i) {
    const size_t e = 0x14 + size_t(i) * 16;
    const uint16_t key_off = le16(b, e), fmt = le16(b, e + 2);
    const uint32_t len = le32(b, e + 4), data_off = le32(b, e + 12);
    const size_t k = size_t(key_table) + key_off, d = size_t(data_table) + data_off;
    if (k >= b.size() || d + len > b.size()) return false;
    std::string key;
    for (size_t p = k; p < b.size() && b[p]; ++p) key += char(b[p]);
    if (fmt == 0x0404 && len >= 4) {
      ints[key] = le32(b, d);
    } else {
      std::string v(reinterpret_cast<const char*>(&b[d]), len);
      while (!v.empty() && v.back() == '\0') v.pop_back();
      strings[key] = v;
    }
  }
  return true;
}

const Sfo& game_sfo() { return g_sfo; }
void set_game_sfo(const Sfo& s) { g_sfo = s; }

}  // namespace vfs
