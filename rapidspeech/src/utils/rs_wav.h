#pragma once

#include "rapidspeech.h"

#include <cstdint>
#include <vector>
/**
 * WAV header， 16bit PCM
 */
struct WaveHeader {
  char chunkId[4];
  uint32_t chunkSize;
  char format[4];
  char subchunk1Id[4];
  uint32_t subchunk1Size;
  uint16_t audioFormat;
  uint16_t numChannels;
  uint32_t sampleRate;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
  char subchunk2Id[4];
  uint32_t subchunk2Size;
};

/**
 * load 16bit PCM WAV convert into float PCM
 * @param filename .wav file path
 * @param data  float
 * @param sample_rate sample rate
 * @return true if success
 */
RS_API bool load_wav_file(const char *filename, std::vector<float> &data,
                          int *sample_rate);

/**
 * Resample mono PCM (float, [-1,1]) to a target sample rate.
 *
 * Uses a Kaiser-windowed sinc (β=8.6, half-width 16 zero-crossings at the
 * lower of the two rates), which gives ~80 dB stop-band attenuation — good
 * enough for ASR/mel feature extraction on common ratios such as 8k→16k,
 * 22050→16000, 44100→16000, 48000→16000.
 *
 * If src_sr == dst_sr the input is copied verbatim. Returns false on
 * non-positive sample rates or empty input.
 */
RS_API bool resample_pcm(const std::vector<float> &in, int src_sr,
                         std::vector<float> &out, int dst_sr);

/**
 * Convenience wrapper: load a WAV file and resample to `target_sample_rate`
 * if its native rate differs. `data` will hold mono float samples in
 * [-1, 1] at `target_sample_rate` on success. The file's native rate is
 * reported back via `src_sample_rate` (may be nullptr).
 */
RS_API bool load_wav_file_resampled(const char *filename,
                                    std::vector<float> &data,
                                    int target_sample_rate,
                                    int *src_sample_rate = nullptr);

void load_cmvn_params(struct gguf_context *ctx_gguf, std::vector<float> &means,
                      std::vector<float> &vars);