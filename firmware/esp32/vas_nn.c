// ============================================================================
//  vas_nn.c — the network, in C.
//
//  Walks the layer table in model_weights.h. Only three op kinds exist (conv,
//  global average pool, linear) because batchnorm was folded into the
//  convolutions at export time, which is what makes a hand-written runtime
//  reasonable here instead of linking TFLite Micro.
//
//  Weights are int16 with a per-output-channel scale; activations stay float32.
//  The S3 has an FPU, four inferences a second is nowhere near its limit, and
//  this avoids an activation calibration pass that would be easy to get subtly
//  wrong. int8 was measured first and rejected: it flipped the gate decision on
//  ~4.4% of real windows once the error had accumulated through ten layers.
// ============================================================================
#include "vas_nn.h"

#include <math.h>
#include <string.h>

// Two ping-pong buffers sized for the largest activation in the graph, which is
// 128 x 10 x 13 = 16640 floats (65 kB). The model strides down to 10x13 before it
// widens to 128 channels specifically to keep this number small — widening at the
// earlier 20x25 resolution would need 32000 floats (128 kB) per buffer, and two of
// those does not fit beside the WiFi stack on an S3. vas_nn_required_arena()
// recomputes this from the layer table at boot and refuses to run if it grew.
static float buf_a[VAS_NN_MAX_ACT];
static float buf_b[VAS_NN_MAX_ACT];

typedef struct { float *d; int c, h, w; } tensor;

static void conv2d(const tensor *src, tensor *dst, const vas_layer *op) {
  const int cin_g = src->c / op->groups;
  const int cout_g = op->cout / op->groups;
  const int k_size = cin_g * op->kh * op->kw;
  const int sH = src->h, sW = src->w;
  const int dH = (sH + 2 * op->ph - op->kh) / op->sh + 1;
  const int dW = (sW + 2 * op->pw - op->kw) / op->sw + 1;

  dst->c = op->cout; dst->h = dH; dst->w = dW;

  for (int oc = 0; oc < op->cout; oc++) {
    const int grp = oc / cout_g;
    const float scale = op->w_s[oc];
    const float bias = op->b[oc];
    const int16_t *wq = op->w_q + (size_t)oc * k_size;
    float *dbase = dst->d + (size_t)oc * dH * dW;

    for (int oy = 0; oy < dH; oy++) {
      const int iy0 = oy * op->sh - op->ph;
      for (int ox = 0; ox < dW; ox++) {
        const int ix0 = ox * op->sw - op->pw;
        float acc = 0.0f;

        for (int ic = 0; ic < cin_g; ic++) {
          const float *splane = src->d + (size_t)(grp * cin_g + ic) * sH * sW;
          const int16_t *krow = wq + (size_t)ic * op->kh * op->kw;
          for (int a = 0; a < op->kh; a++) {
            const int iy = iy0 + a;
            if (iy < 0 || iy >= sH) continue;
            for (int b = 0; b < op->kw; b++) {
              const int ix = ix0 + b;
              if (ix < 0 || ix >= sW) continue;
              acc += splane[iy * sW + ix] * (float)krow[a * op->kw + b];
            }
          }
        }
        acc = acc * scale + bias;
        dbase[oy * dW + ox] = (op->relu && acc < 0.0f) ? 0.0f : acc;
      }
    }
  }
}

static void global_avg_pool(const tensor *src, tensor *dst) {
  const int plane = src->h * src->w;
  for (int c = 0; c < src->c; c++) {
    const float *p = src->d + (size_t)c * plane;
    float acc = 0.0f;
    for (int i = 0; i < plane; i++) acc += p[i];
    dst->d[c] = acc / (float)plane;
  }
  dst->c = src->c; dst->h = 1; dst->w = 1;
}

static void linear(const tensor *src, tensor *dst, const vas_layer *op) {
  for (int o = 0; o < op->cout; o++) {
    const int16_t *wq = op->w_q + (size_t)o * op->cin;
    float acc = 0.0f;
    for (int i = 0; i < op->cin; i++) acc += src->d[i] * (float)wq[i];
    acc = acc * op->w_s[o] + op->b[o];
    dst->d[o] = (op->relu && acc < 0.0f) ? 0.0f : acc;
  }
  dst->c = op->cout; dst->h = 1; dst->w = 1;
}

void vas_nn_forward(const float *feat, float *logits) {
  tensor a = { buf_a, 1, VAS_N_BINS, VAS_N_FRAMES };
  tensor b = { buf_b, 0, 0, 0 };
  memcpy(buf_a, feat, sizeof(float) * VAS_N_BINS * VAS_N_FRAMES);

  tensor *src = &a, *dst = &b;
  for (size_t i = 0; i < VAS_N_LAYERS; i++) {
    const vas_layer *op = &VAS_LAYERS[i];
    switch (op->kind) {
      case VAS_OP_CONV:   conv2d(src, dst, op); break;
      case VAS_OP_GAP:    global_avg_pool(src, dst); break;
      case VAS_OP_LINEAR: linear(src, dst, op); break;
    }
    tensor *t = src; src = dst; dst = t;   // ping-pong
  }
  memcpy(logits, src->d, sizeof(float) * VAS_N_CLASSES);
}

void vas_nn_softmax(const float *logits, float *probs) {
  float max = logits[0] / VAS_TEMPERATURE;
  float z[VAS_N_CLASSES];
  for (int i = 0; i < VAS_N_CLASSES; i++) {
    z[i] = logits[i] / VAS_TEMPERATURE;      // calibration fitted on validation
    if (z[i] > max) max = z[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < VAS_N_CLASSES; i++) { z[i] = expf(z[i] - max); sum += z[i]; }
  for (int i = 0; i < VAS_N_CLASSES; i++) probs[i] = z[i] / sum;
}

int vas_nn_argmax(const float *probs) {
  int best = 0;
  for (int i = 1; i < VAS_N_CLASSES; i++) if (probs[i] > probs[best]) best = i;
  return best;
}

size_t vas_nn_required_arena(void) {
  size_t need = (size_t)VAS_N_BINS * VAS_N_FRAMES;   // the input tensor
  int c = 1, h = VAS_N_BINS, w = VAS_N_FRAMES;

  for (size_t i = 0; i < VAS_N_LAYERS; i++) {
    const vas_layer *op = &VAS_LAYERS[i];
    switch (op->kind) {
      case VAS_OP_CONV:
        h = (h + 2 * op->ph - op->kh) / op->sh + 1;
        w = (w + 2 * op->pw - op->kw) / op->sw + 1;
        c = op->cout;
        break;
      case VAS_OP_GAP:
        h = 1; w = 1;
        break;
      case VAS_OP_LINEAR:
        c = op->cout; h = 1; w = 1;
        break;
    }
    const size_t sz = (size_t)c * h * w;
    if (sz > need) need = sz;
  }
  return need;
}
