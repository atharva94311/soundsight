#include "vas_dsp.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
//  Tables, built once at boot.
//
//  The filterbank is stored sparsely (first bin + run length per mel band)
//  because it is ~97% zeros: storing it dense would cost 40 * 513 * 4 = 82 kB of
//  RAM to multiply mostly by zero.
// ---------------------------------------------------------------------------
static float win[VAS_WIN_LENGTH];
static float fft_cos[VAS_N_FFT / 2];
static float fft_sin[VAS_N_FFT / 2];
static uint16_t bitrev[VAS_N_FFT];

#define VAS_FB_MAX 96            // widest mel band, in FFT bins (checked at init)
static float fb_w[VAS_N_BINS][VAS_FB_MAX];
static uint16_t fb_start[VAS_N_BINS];
static uint16_t fb_len[VAS_N_BINS];

static float re[VAS_N_FFT];
static float im[VAS_N_FFT];

static inline float hz_to_mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
static inline float mel_to_hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

void vas_dsp_init(void) {
  for (int i = 0; i < VAS_WIN_LENGTH; i++)
    win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / VAS_WIN_LENGTH);

  int bits = 0;
  while ((1 << bits) < VAS_N_FFT) bits++;
  for (int i = 0; i < VAS_N_FFT; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    bitrev[i] = (uint16_t)r;
  }
  // Twiddles for the largest stage; smaller stages index this table by stride.
  for (int j = 0; j < VAS_N_FFT / 2; j++) {
    float ang = -2.0f * (float)M_PI * j / VAS_N_FFT;
    fft_cos[j] = cosf(ang);
    fft_sin[j] = sinf(ang);
  }

  float m0 = hz_to_mel(VAS_FMIN), m1 = hz_to_mel(VAS_FMAX);
  for (int m = 0; m < VAS_N_BINS; m++) {
    float lo = mel_to_hz(m0 + (m1 - m0) * (m + 0) / (VAS_N_BINS + 1)) * VAS_N_FFT / VAS_SAMPLE_RATE;
    float mid = mel_to_hz(m0 + (m1 - m0) * (m + 1) / (VAS_N_BINS + 1)) * VAS_N_FFT / VAS_SAMPLE_RATE;
    float hi = mel_to_hz(m0 + (m1 - m0) * (m + 2) / (VAS_N_BINS + 1)) * VAS_N_FFT / VAS_SAMPLE_RATE;

    int k0 = (int)lo, k1 = (int)hi + 2;
    if (k0 < 0) k0 = 0;
    if (k1 > VAS_N_SPEC) k1 = VAS_N_SPEC;

    int n = 0, start = -1;
    for (int k = k0; k < k1; k++) {
      float v = 0.0f;
      if (mid > lo && k > lo && k <= mid) v = (k - lo) / (mid - lo);
      else if (hi > mid && k > mid && k < hi) v = (hi - k) / (hi - mid);
      if (v <= 0.0f) { if (start < 0) continue; else if (n > 0) break; }
      if (start < 0) start = k;
      if (n < VAS_FB_MAX) fb_w[m][n++] = v;
    }
    fb_start[m] = (uint16_t)(start < 0 ? 0 : start);
    fb_len[m] = (uint16_t)n;
  }
}

/** In-place radix-2 FFT, same structure as the JS one in mel.js. */
static void fft(float *xr, float *xi) {
  for (int i = 0; i < VAS_N_FFT; i++) {
    int j = bitrev[i];
    if (j > i) {
      float t = xr[i]; xr[i] = xr[j]; xr[j] = t;
      t = xi[i]; xi[i] = xi[j]; xi[j] = t;
    }
  }
  for (int len = 2; len <= VAS_N_FFT; len <<= 1) {
    int half = len >> 1;
    int stride = VAS_N_FFT / len;
    for (int i = 0; i < VAS_N_FFT; i += len) {
      for (int j = 0; j < half; j++) {
        float c = fft_cos[j * stride], s = fft_sin[j * stride];
        int a = i + j, b = a + half;
        float tr = xr[b] * c - xi[b] * s;
        float ti = xr[b] * s + xi[b] * c;
        xr[b] = xr[a] - tr; xi[b] = xi[a] - ti;
        xr[a] += tr;        xi[a] += ti;
      }
    }
  }
}

void vas_logmel(const float *samples, float *out) {
  for (int t = 0; t < VAS_N_FRAMES; t++) {
    const int off = t * VAS_HOP_LENGTH;

    memset(re, 0, sizeof(re));
    memset(im, 0, sizeof(im));
    for (int i = 0; i < VAS_WIN_LENGTH; i++) re[i] = samples[off + i] * win[i];
    fft(re, im);

    for (int m = 0; m < VAS_N_BINS; m++) {
      const int s = fb_start[m], n = fb_len[m];
      float acc = 0.0f;
      for (int k = 0; k < n; k++) {
        const int b = s + k;
        acc += (re[b] * re[b] + im[b] * im[b]) * fb_w[m][k];
      }
      // Standardise here so the network sees exactly what it trained on.
      out[m * VAS_N_FRAMES + t] = (logf(acc + VAS_LOG_FLOOR) - VAS_NORM_MEAN) / VAS_NORM_STD;
    }
  }
}

float vas_rms(const float *x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; i++) acc += x[i] * x[i];
  return sqrtf(acc / (float)n);
}
