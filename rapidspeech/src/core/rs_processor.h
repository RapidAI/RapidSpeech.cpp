#pragma once

#include "frontend/audio_processor.h"
#include "ggml-backend.h"
#include "rs_model.h"
#include <deque>
#include <memory>
#include <string>
#include <vector>
#define SENSE_VOICE_MAX_GRAPH_SIZE 8196
/**
 * Simple thread-safe-ish circular buffer for raw PCM samples.
 */
class CircularBuffer {
public:
  void Push(const float *data, size_t size);
  std::vector<float> Pop(size_t size);
  size_t Size() const;

private:
  std::deque<float> buffer_;
};

/**
 * RSProcessor orchestrates the entire speech processing pipeline:
 * Audio Buffer -> Feature Extraction -> Model Encoding -> Model Decoding ->
 * Text/Audio Output
 *
 * Supports both ASR (text output) and TTS (audio output) tasks.
 */
class RSProcessor {
public:
  /**
   * Constructor
   * @param model Shared pointer to the architecture-specific model
   * @param sched Backend scheduler used for inference
   */
  RSProcessor(std::shared_ptr<ISpeechModel> model, ggml_backend_sched_t sched);

  // --- ASR methods ---

  /**
   * Pushes raw PCM audio samples into the internal buffer.
   */
  void PushAudio(const float *data, size_t size);

  /**
   * Updates the CMVN (Mean and Variance) parameters for the audio frontend.
   */
  void SetCMVN(const std::vector<float> &means, const std::vector<float> &vars);

  /**
   * Sets the user input prompt for the LLM decoder (FunASRNano only).
   */
  void SetUserInputPrompt(const std::string &prompt);

  /**
   * Executes one iteration of the processing pipeline.
   * @return 0: No new results, 1: New text/audio output available, -1: Error
   */
  int Process();

  /**
   * Re-run Decode only (skip Encode), used for 2-pass rescoring.
   * @return 0: No output, 1: Has output, -1: Error
   */
  int DecodeOnly();

  /**
   * Returns the accumulated text result.
   */
  std::string GetTextResult();

  // --- TTS methods ---

  /**
   * Set TTS generation parameters.
   */
  void SetTTSParams(const char *instruct, const char *language, int seed);

  /**
   * Push text for TTS synthesis.
   * @return 0 on success, -1 on error
   */
  int PushText(const char *text, const char *language = nullptr);

  /**
   * Push reference audio for voice cloning (optional, TTS only).
   * @return 0 on success, -1 on error
   */
  int PushReferenceAudio(const float *samples, int n_samples, int sample_rate);

  /**
   * Push reference text for voice cloning (optional, TTS only).
   * Must be called before ProcessTTS if using reference audio.
   * @return 0 on success, -1 on error
   */
  int PushReferenceText(const char *ref_text);

  /**
   * Run one TTS processing step.
   * First call runs encoder + duration + flow decoder.
   * Subsequent calls run vocoder on mel chunks.
   * @return 1: Audio available, 0: Done, -1: Error
   */
  int ProcessTTS();

  /**
   * Get synthesized audio output.
   * @param out_data Pointer to internal buffer (do not free)
   * @return Number of samples available, 0 if none
   */
  int GetAudioOutput(float **out_data);

  // --- Internal API for C Bindings ---

  size_t GetAudioBufferSize() const { return audio_buffer_.Size(); }

  int GetAudioSampleRate() const {
    return model_ ? model_->GetMeta().audio_sample_rate : 16000;
  }

  bool IsValid() const {
    return model_ != nullptr && state_ != nullptr && sched_ != nullptr;
  }

  std::string GetArchName() const {
    return model_ ? model_->GetMeta().arch_name : std::string();
  }

  /**
   * Reset processor state for a new utterance.
   */
  void Reset();

private:
  std::shared_ptr<ISpeechModel> model_;
  std::shared_ptr<RSState> state_;
  std::unique_ptr<AudioProcessor> audio_proc_;

  ggml_backend_sched_t sched_;

  CircularBuffer audio_buffer_;
  std::string text_accumulator_;

  int last_token_id_ = -1;
  int chunk_size_samples_ = 8000; // 500ms chunks

  // TTS parameters
  std::string tts_instruct_ = "male";
  std::string tts_language_ = "English";
  int tts_seed_ = 42;

  // TTS state
  bool tts_encoded_ = false;
  bool tts_decoded_ = false;
  std::vector<float> tts_audio_buf_;
  int tts_audio_read_pos_ = 0;
};
