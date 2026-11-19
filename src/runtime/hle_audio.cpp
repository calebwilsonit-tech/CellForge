// cellAudio and cellAudioOut HLE on top of audio::Mixer.
// Struct layouts and constants follow the public cellAudio documentation.

#include <algorithm>
#include <bit>
#include <cstring>

#include "audio.h"
#include "hle.h"
#include "lv2.h"

namespace hle {
namespace {

using audio::Mixer;
using lv2::Kernel;

constexpr uint32_t CELL_OK = 0;
constexpr uint32_t CELL_AUDIO_ERROR_ALREADY_INIT = 0x80310701;
constexpr uint32_t CELL_AUDIO_ERROR_NOT_INIT = 0x80310703;
constexpr uint32_t CELL_AUDIO_ERROR_PARAM = 0x80310704;
constexpr uint32_t CELL_AUDIO_ERROR_PORT_FULL = 0x80310705;
constexpr uint32_t CELL_AUDIO_ERROR_PORT_ALREADY_RUN = 0x80310706;
constexpr uint32_t CELL_AUDIO_ERROR_PORT_NOT_OPEN = 0x80310707;
constexpr uint32_t CELL_AUDIO_ERROR_PORT_NOT_RUN = 0x80310708;
constexpr uint32_t CELL_AUDIO_ERROR_EVENT_QUEUE = 0x8031070d;
constexpr uint32_t CELL_AUDIO_ERROR_TAG_NOT_FOUND = 0x8031070f;
constexpr uint32_t CELL_AUDIO_STATUS_CLOSE = 0x1010, CELL_AUDIO_STATUS_READY = 1, CELL_AUDIO_STATUS_RUN = 2;
constexpr uint64_t CELL_AUDIO_PORTATTR_INITLEVEL = 0x1000;

Kernel& K() { return Kernel::get(); }
uint32_t a32(ppu::Context& c, int n) { return uint32_t(c.r[3 + n]); }
void ret(ppu::Context& c, uint64_t v) { c.r[3] = v; }

bool g_init = false;
uint64_t g_next_key = 0x80004D494F323200ull;  // distinct from keys games choose themselves

audio::Port* port_of(ppu::Context&, uint32_t n, bool must_be_open = true) {
  if (n >= audio::kMaxPorts) return nullptr;
  audio::Port* p = &Mixer::get().ports[n];
  return (!must_be_open || p->open) ? p : nullptr;
}

void cellAudioInit(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (g_init) return ret(c, CELL_AUDIO_ERROR_ALREADY_INIT);
  if (!mx.index_area) mx.index_area = K().hle.alloc(8 * audio::kMaxPorts, 128);
  for (auto& p : mx.ports) p = audio::Port{};
  mx.notify_keys.clear();
  g_init = true;
  if (!mx.started()) {
    auto out = audio::create_wasapi_output();
    if (!out) K().logf("cellAudio: no host audio device, running silent");
    mx.start(std::move(out));
  }
  ret(c, CELL_OK);
}
void cellAudioQuit(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  for (auto& p : mx.ports) {
    if (p.addr) K().heap.free(p.addr);
    p = audio::Port{};
  }
  g_init = false;
  ret(c, CELL_OK);
}

void cellAudioPortOpen(ppu::Context& c) {
  // (const CellAudioPortParam* param, u32* portNum)
  // param: u64 nChannel, u64 nBlock, u64 attr, float level
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  const uint32_t prm = a32(c, 0);
  if (!prm || !a32(c, 1)) return ret(c, CELL_AUDIO_ERROR_PARAM);
  const uint64_t channels = K().r64(prm), blocks = K().r64(prm + 8), attr = K().r64(prm + 16);
  const float level = std::bit_cast<float>(K().r32(prm + 24));
  if ((channels != 2 && channels != 8) || (blocks != 8 && blocks != 16 && blocks != 32))
    return ret(c, CELL_AUDIO_ERROR_PARAM);
  for (uint32_t n = 0; n < audio::kMaxPorts; ++n) {
    audio::Port& p = mx.ports[n];
    if (p.open) continue;
    p = audio::Port{};
    p.channels = uint32_t(channels);
    p.blocks = uint32_t(blocks);
    p.size = p.channels * p.blocks * audio::kBlockFrames * 4;
    p.addr = K().heap.alloc(p.size, 0x100);
    if (!p.addr) return ret(c, CELL_AUDIO_ERROR_PORT_FULL);
    p.level = (attr & CELL_AUDIO_PORTATTR_INITLEVEL) ? level : 1.0f;
    p.open = true;
    mx.set_read_index(n, 0);
    K().w32(a32(c, 1), n);
    return ret(c, CELL_OK);
  }
  ret(c, CELL_AUDIO_ERROR_PORT_FULL);
}

void cellAudioGetPortConfig(ppu::Context& c) {
  // CellAudioPortConfig: u32 readIndexAddr, u32 status, u64 nChannel, u64 nBlock, u32 portSize, u32 portAddr
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  const uint32_t n = a32(c, 0), cfg = a32(c, 1);
  if (n >= audio::kMaxPorts || !cfg) return ret(c, CELL_AUDIO_ERROR_PARAM);
  const audio::Port& p = mx.ports[n];
  K().w32(cfg + 0, mx.index_area + 8 * n);
  K().w32(cfg + 4, !p.open ? CELL_AUDIO_STATUS_CLOSE : p.running ? CELL_AUDIO_STATUS_RUN : CELL_AUDIO_STATUS_READY);
  K().w64(cfg + 8, p.channels);
  K().w64(cfg + 16, p.blocks);
  K().w32(cfg + 24, p.size);
  K().w32(cfg + 28, p.addr);
  ret(c, CELL_OK);
}

void cellAudioPortStart(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  if (p->running) return ret(c, CELL_AUDIO_ERROR_PORT_ALREADY_RUN);
  p->running = true;
  ret(c, CELL_OK);
}
void cellAudioPortStop(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  if (!p->running) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_RUN);
  p->running = false;
  ret(c, CELL_OK);
}
void cellAudioPortClose(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  if (p->addr) K().heap.free(p->addr);
  *p = audio::Port{};
  ret(c, CELL_OK);
}

void cellAudioGetPortTimestamp(ppu::Context& c) {
  // (u32 port, u64 tag, u64* stamp)
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  const uint64_t tag = c.r[4];
  for (uint32_t b = 0; b < p->blocks; ++b)
    if (p->block_tag[b] == tag && p->block_time[b]) {
      K().w64(a32(c, 2), p->block_time[b]);
      return ret(c, CELL_OK);
    }
  ret(c, CELL_AUDIO_ERROR_TAG_NOT_FOUND);
}
void cellAudioGetPortBlockTag(ppu::Context& c) {
  // (u32 port, u64 blockNo, u64* tag)
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  if (c.r[4] >= p->blocks) return ret(c, CELL_AUDIO_ERROR_PARAM);
  K().w64(a32(c, 2), p->block_tag[c.r[4]]);
  ret(c, CELL_OK);
}

void cellAudioSetPortLevel(ppu::Context& c) {
  // (u32 port, float level) - the float arrives in f1
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  if (c.f[1] < 0) return ret(c, CELL_AUDIO_ERROR_PARAM);
  p->level = float(c.f[1]);
  ret(c, CELL_OK);
}

void cellAudioCreateNotifyEventQueue(ppu::Context& c) {
  // (u32* id, u64* key)
  if (!a32(c, 0) || !a32(c, 1)) return ret(c, CELL_AUDIO_ERROR_PARAM);
  uint32_t id = 0;
  const uint64_t key = g_next_key++;
  if (lv2::event_queue_create(32, key, &id) != lv2::CELL_OK) return ret(c, CELL_AUDIO_ERROR_EVENT_QUEUE);
  K().w32(a32(c, 0), id);
  K().w64(a32(c, 1), key);
  ret(c, CELL_OK);
}
void cellAudioSetNotifyEventQueue(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  const uint64_t key = c.r[3];
  if (std::find(mx.notify_keys.begin(), mx.notify_keys.end(), key) == mx.notify_keys.end()) mx.notify_keys.push_back(key);
  ret(c, CELL_OK);
}
void cellAudioRemoveNotifyEventQueue(ppu::Context& c) {
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  auto& keys = mx.notify_keys;
  keys.erase(std::remove(keys.begin(), keys.end(), c.r[3]), keys.end());
  ret(c, CELL_OK);
}

// Mixes extra samples into the block the mixer will play next.
void add_data(ppu::Context& c, uint32_t src_channels) {
  // (u32 port, const float* src, u32 samples, float volume in f1)
  Mixer& mx = Mixer::get();
  std::lock_guard<std::mutex> lk(mx.m);
  if (!g_init) return ret(c, CELL_AUDIO_ERROR_NOT_INIT);
  audio::Port* p = port_of(c, a32(c, 0));
  if (!p) return ret(c, CELL_AUDIO_ERROR_PORT_NOT_OPEN);
  const uint32_t src = a32(c, 1), frames = std::min<uint32_t>(a32(c, 2), audio::kBlockFrames);
  const float vol = float(c.f[1]);
  const uint32_t block = uint32_t((mx.read_index(a32(c, 0)) + 1) % p->blocks);
  uint8_t* mem = K().base();
  const uint32_t dst = p->addr + block * p->channels * audio::kBlockFrames * 4;
  for (uint32_t f = 0; f < frames; ++f)
    for (uint32_t ch = 0; ch < std::min(src_channels, p->channels); ++ch) {
      const uint32_t s = src + (f * src_channels + ch) * 4, d = dst + (f * p->channels + ch) * 4;
      const float v = std::bit_cast<float>(ppu::ld32(mem, d)) + std::bit_cast<float>(ppu::ld32(mem, s)) * vol;
      ppu::st32(mem, d, std::bit_cast<uint32_t>(v));
    }
  ret(c, CELL_OK);
}
void cellAudioAddData(ppu::Context& c) { add_data(c, Mixer::get().ports[a32(c, 0) & 3].channels); }
void cellAudioAdd2chData(ppu::Context& c) { add_data(c, 2); }
void cellAudioAdd6chData(ppu::Context& c) { add_data(c, 6); }

void audio_ok(ppu::Context& c) { ret(c, CELL_OK); }

// ---- cellAudioOut (cellSysutil) ----
void cellAudioOutGetSoundAvailability(ppu::Context& c) { ret(c, 2); }  // stereo LPCM
void cellAudioOutGetNumberOfDevice(ppu::Context& c) { ret(c, 1); }
void cellAudioOutGetState(ppu::Context& c) {
  // (u32 audioOut, u32 deviceIndex, CellAudioOutState* state): u8 state, u8 encoder, u8 reserved[6],
  // u32 downMixer, CellAudioOutSoundMode soundMode { u8 type, u8 channel, u8 fs, u8 reserved, u32 layout }
  const uint32_t a = a32(c, 2);
  std::memset(K().base() + a, 0, 20);
  K().base()[a + 0] = 0;     // enabled
  K().base()[a + 12] = 0;    // LPCM
  K().base()[a + 13] = 2;    // 2 channels
  K().base()[a + 14] = 4;    // 48 kHz
  ret(c, CELL_OK);
}

}  // namespace

void register_audio_modules() {
#define R(fn) register_function("cellAudio", #fn, fn)
  R(cellAudioInit); R(cellAudioQuit); R(cellAudioPortOpen); R(cellAudioGetPortConfig); R(cellAudioPortStart);
  R(cellAudioPortStop); R(cellAudioPortClose); R(cellAudioGetPortTimestamp); R(cellAudioGetPortBlockTag);
  R(cellAudioSetPortLevel); R(cellAudioCreateNotifyEventQueue); R(cellAudioSetNotifyEventQueue);
  R(cellAudioRemoveNotifyEventQueue); R(cellAudioAddData); R(cellAudioAdd2chData); R(cellAudioAdd6chData);
#undef R
  register_function("cellAudio", "cellAudioCreateNotifyEventQueueEx", cellAudioCreateNotifyEventQueue);
  register_function("cellAudio", "cellAudioSetNotifyEventQueueEx", cellAudioSetNotifyEventQueue);
  register_function("cellAudio", "cellAudioRemoveNotifyEventQueueEx", cellAudioRemoveNotifyEventQueue);
  for (const char* n : {"cellAudioMiscSetAccessoryVolume", "cellAudioSendAck", "cellAudioSetPersonalDevice",
                        "cellAudioUnsetPersonalDevice", "cellAudioAddData"})
    if (!lookup("cellAudio", n)) register_function("cellAudio", n, audio_ok);
  register_function("cellSysutil", "cellAudioOutGetSoundAvailability", cellAudioOutGetSoundAvailability);
  register_function("cellSysutil", "cellAudioOutGetSoundAvailability2", cellAudioOutGetSoundAvailability);
  register_function("cellSysutil", "cellAudioOutGetNumberOfDevice", cellAudioOutGetNumberOfDevice);
  register_function("cellSysutil", "cellAudioOutGetState", cellAudioOutGetState);
  register_function("cellSysutil", "cellAudioOutConfigure", audio_ok);
}

}  // namespace hle
