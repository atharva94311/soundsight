// ============================================================================
//  vas_dsp.h — the audio front-end, in C.
//
//  Third implementation of the same maths as ml/vas_ml/features.py and
//  vas3d/js/audio/mel.js. Same window, same mel filterbank, same log floor. If
//  this drifts from the Python version the model will still run and still look
//  confident, it will just be wrong — so the constants come from the generated
//  model_weights.h rather than being retyped here.
//
//  Fixed-size buffers, no malloc: everything is sized at compile time from the
//  model geometry so the memory cost is visible in the map file.
// ============================================================================
#pragma once

#include <stdint.h>
#include "model_weights.h"

#define VAS_SAMPLE_RATE 16000
#define VAS_WIN_LENGTH  640      // 40 ms
#define VAS_HOP_LENGTH  320      // 20 ms
#define VAS_N_FFT       1024
#define VAS_N_SPEC      (VAS_N_FFT / 2 + 1)
#define VAS_N_SAMPLES   (VAS_SAMPLE_RATE)   // 1 s inference window
#define VAS_LOG_FLOOR   1e-6f
#define VAS_FMIN        20.0f
#define VAS_FMAX        7800.0f

#ifdef __cplusplus
extern "C" {
#endif

/** Build the window and mel filterbank. Call once at boot, before vas_logmel. */
void vas_dsp_init(void);

/**
 * Log-mel spectrogram of one 1-second window.
 *
 * @param samples  VAS_N_SAMPLES floats in [-1, 1]
 * @param out      VAS_N_BINS * VAS_N_FRAMES floats, row-major (bin-major),
 *                 already standardised with VAS_NORM_MEAN / VAS_NORM_STD —
 *                 i.e. exactly what the network expects.
 */
void vas_logmel(const float *samples, float *out);

/** RMS of a buffer; used for the "is anything happening" gate before inference. */
float vas_rms(const float *x, int n);

#ifdef __cplusplus
}
#endif
