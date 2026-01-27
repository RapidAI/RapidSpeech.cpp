#include "utils/rs_wav.h"
#include "gguf.h"
#include "rs_log.h"
#include <cstring>
#include <fstream>
#include <iostream>

bool load_wav_file(const char* filename, std::vector<float>& data, int* sample_rate) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  WaveHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(WaveHeader));

  // 基础校验
  if (std::strncmp(header.chunkId, "RIFF", 4) != 0 || std::strncmp(header.format, "WAVE", 4) != 0) {
    std::cerr << "[rs_wav] 错误：非法的 RIFF/WAVE 文件。" << std::endl;
    return false;
  }

  if (header.bitsPerSample != 16) {
    std::cerr << "[rs_wav] 错误：仅支持 16位 PCM WAV 文件。" << std::endl;
    return false;
  }

  *sample_rate = static_cast<int>(header.sampleRate);
    
  // 计算采样点数量
  int num_samples = header.subchunk2Size / (header.numChannels * (header.bitsPerSample / 8));
  data.resize(num_samples);

  // 读取 16位数据并归一化到 float [-1.0, 1.0]
  for (int i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
    data[i] = static_cast<float>(sample) / 32768.0f;
        
    // 如果是双声道，为了简单起见跳过第二个声道
    if (header.numChannels > 1) {
      file.seekg(sizeof(int16_t), std::ios::cur);
    }
  }

  return true;
}

static float DEFAULT_CMVN_MEANS[560];

static float DEFAULT_CMVN_VARS[560];

/**
 * Helper to load CMVN from GGUF or fallback to defaults.
 * @param ctx_gguf The loaded GGUF context.
 * @param means Target vector for mean values.
 * @param vars Target vector for variance values.
 */
void load_cmvn_params(struct gguf_context * ctx_gguf, std::vector<float>& means, std::vector<float>& vars) {
  bool loaded = false;

  // 1. Try to load from GGUF KV
  int key_means = gguf_find_key(ctx_gguf, "model.cmvn_means");
  int key_vars  = gguf_find_key(ctx_gguf, "model.cmvn_vars");

  if (key_means != -1 && key_vars != -1) {
    const float * data_means = (const float *) gguf_get_arr_data(ctx_gguf, key_means);
    const float * data_vars  = (const float *) gguf_get_arr_data(ctx_gguf, key_vars);
    int n_means = gguf_get_arr_n(ctx_gguf, key_means);
    int n_vars  = gguf_get_arr_n(ctx_gguf, key_vars);

    if (n_means == 560 && n_vars == 560) {
      means.assign(data_means, data_means + 560);
      vars.assign(data_vars, data_vars + 560);
      RS_LOG_INFO("CMVN parameters loaded from GGUF KV metadata.");
      loaded = true;
    } else {
      RS_LOG_WARN("CMVN metadata size mismatch (expected 560, got %d). Falling back.", n_means);
    }
  }

  // 2. Fallback to hardcoded defaults if GGUF doesn't have them
  if (!loaded) {
    RS_LOG_INFO("Using hardcoded default CMVN parameters.");
    means.assign(DEFAULT_CMVN_MEANS, DEFAULT_CMVN_MEANS + 560);
    vars.assign(DEFAULT_CMVN_VARS, DEFAULT_CMVN_VARS + 560);
  }
}