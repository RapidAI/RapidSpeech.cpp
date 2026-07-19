//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
// NOTE (RapidSpeech.cpp): trimmed from the upstream ten-vad aed_st.h.
// Only the configuration macros needed by coeff.h are kept; the upstream
// AUP_MODULE_AIVAD class and Aed_St struct depend on onnxruntime_c_api.h,
// which this project does not use (inference runs through ggml instead,
// see rapidspeech/src/arch/tenvad.cpp).
//
#ifndef __AED_ST_H__
#define __AED_ST_H__

#define AUP_AED_FS (16000)
#define AUP_AED_MAX_IN_BUFF_SIZE (256)
#define AUP_AED_POWER_SPCTR_NORMALIZER (9.3132e-10f)  // = 1/(32768^2)
#define AUP_AED_OUTPUT_SMOOTH_FILTER_LEN (10)         // 160ms

#define AUP_AED_MEL_FILTER_BANK_NUM (40)
#define AUP_AED_LOOKAHEAD_NFRM (1)
#define AUP_AED_CONTEXT_WINDOW_LEN (3)  // context window length of AIVAD
#define AUP_AED_FEA_LEN \
  (AUP_AED_MEL_FILTER_BANK_NUM + 1)  // feature length of AIVAD

#define AUP_AED_PITCH_EST_USE_LPC (1)
#define AUP_AED_PITCH_EST_PROCFS (4000)
#if AUP_AED_PITCH_EST_PROCFS == 2000
#define AUP_AED_PITCH_EST_DEFAULT_VOICEDTHR (0.45f)
#else
#define AUP_AED_PITCH_EST_DEFAULT_VOICEDTHR (0.4f)
#endif

#endif  // __AED_ST_H__
