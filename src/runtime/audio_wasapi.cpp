// Host audio output through WASAPI (shared mode, default render device). Windows
// converts our 48 kHz float stereo to the device format, so no resampler is needed.

#include "audio.h"

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <mmreg.h>

#include <algorithm>
#include <thread>

namespace audio {
namespace {

class WasapiOutput final : public Output {
 public:
  ~WasapiOutput() override {
    if (client_) client_->Stop();
    if (render_) render_->Release();
    if (client_) client_->Release();
    if (device_) device_->Release();
    if (com_) CoUninitialize();
  }

  bool init() {
    // COM on whichever thread first writes (the mixer thread); init itself only probes.
    return true;
  }

  // Never blocks: the mixer keeps time on its own precise timer and steers by
  // queued_frames(). Blocking here used to tie the game's audio events to WASAPI's
  // 10 ms engine quantum, delivering them in uneven pairs.
  bool paced() const override { return false; }
  bool active() const override { return ready_; }

  int64_t queued_frames() const override {
    if (!ready_) return -1;
    UINT32 padding = 0;
    if (FAILED(client_->GetCurrentPadding(&padding))) return -1;
    return padding;
  }

  void write(const float* stereo, uint32_t frames) override {
    if (!ready_ && !failed_) open();
    if (!ready_) return;  // no device: the mixer's own timer keeps the game at speed
    UINT32 padding = 0;
    if (FAILED(client_->GetCurrentPadding(&padding))) return;
    // If the device has fallen this far behind, drop the block rather than wait.
    const uint32_t n = std::min(buffer_frames_ - padding, frames);
    if (n == 0) return;
    BYTE* data = nullptr;
    if (FAILED(render_->GetBuffer(n, &data))) return;
    std::memcpy(data, stereo, size_t(n) * 2 * sizeof(float));
    render_->ReleaseBuffer(n, 0);
  }

 private:
  void open() {
    com_ = SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
    IMMDeviceEnumerator* en = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                reinterpret_cast<void**>(&en)))) {
      failed_ = true;
      return;
    }
    const HRESULT hr = en->GetDefaultAudioEndpoint(eRender, eConsole, &device_);
    en->Release();
    if (FAILED(hr) || FAILED(device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                                               reinterpret_cast<void**>(&client_)))) {
      failed_ = true;
      return;
    }
    WAVEFORMATEXTENSIBLE fmt{};
    fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    fmt.Format.nChannels = 2;
    fmt.Format.nSamplesPerSec = kSampleRate;
    fmt.Format.wBitsPerSample = 32;
    fmt.Format.nBlockAlign = 2 * 4;
    fmt.Format.nAvgBytesPerSec = kSampleRate * 8;
    fmt.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    fmt.Samples.wValidBitsPerSample = 32;
    fmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
    fmt.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
    const REFERENCE_TIME buffer = 200 * 10000;  // 200 ms of room; the mixer keeps ~43 ms queued
    if (FAILED(client_->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                   AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, buffer,
                                   0, &fmt.Format, nullptr)) ||
        FAILED(client_->GetBufferSize(&buffer_frames_)) ||
        FAILED(client_->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&render_)))) {
      failed_ = true;
      return;
    }
    // Start with ~32 ms of silence queued so the first engine passes don't underrun
    // while the mixer's queue level settles.
    const UINT32 prefill = std::min<UINT32>(buffer_frames_, 6 * kBlockFrames);
    BYTE* data = nullptr;
    if (prefill && SUCCEEDED(render_->GetBuffer(prefill, &data))) render_->ReleaseBuffer(prefill, AUDCLNT_BUFFERFLAGS_SILENT);
    if (FAILED(client_->Start())) {
      failed_ = true;
      return;
    }
    ready_ = true;
  }

  bool com_ = false, ready_ = false, failed_ = false;
  IMMDevice* device_ = nullptr;
  IAudioClient* client_ = nullptr;
  IAudioRenderClient* render_ = nullptr;
  UINT32 buffer_frames_ = 0;
};

}  // namespace

std::unique_ptr<Output> create_wasapi_output() {
  auto o = std::make_unique<WasapiOutput>();
  if (!o->init()) return nullptr;
  return o;
}

}  // namespace audio

#else

namespace audio {
std::unique_ptr<Output> create_wasapi_output() { return nullptr; }
}  // namespace audio

#endif
