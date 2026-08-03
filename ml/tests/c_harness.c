// Host-side driver for the firmware DSP + NN, used by test_c_parity.py.
//
//   c_harness <signals.bin> <out.bin>
//
// signals.bin:  int32 count, then count * VAS_N_SAMPLES float32
// out.bin:      per signal, VAS_N_BINS*VAS_N_FRAMES float32 features,
//               then VAS_N_CLASSES float32 logits, then VAS_N_CLASSES float32 probs
#include <stdio.h>
#include <stdlib.h>

#include "vas_dsp.h"
#include "vas_nn.h"

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <in> <out>\n", argv[0]); return 2; }

  FILE *fi = fopen(argv[1], "rb");
  if (!fi) { perror("open in"); return 2; }
  int32_t count = 0;
  if (fread(&count, sizeof(count), 1, fi) != 1) { fprintf(stderr, "bad header\n"); return 2; }

  vas_dsp_init();

  size_t need = vas_nn_required_arena();
  if (need > VAS_NN_MAX_ACT) {
    fprintf(stderr, "arena too small: need %zu have %d\n", need, VAS_NN_MAX_ACT);
    return 3;
  }
  fprintf(stderr, "arena: need %zu floats (%.1f kB), have %d\n",
          need, need * 4 / 1024.0, VAS_NN_MAX_ACT);

  FILE *fo = fopen(argv[2], "wb");
  if (!fo) { perror("open out"); return 2; }

  float *samples = malloc(sizeof(float) * VAS_N_SAMPLES);
  float *feat = malloc(sizeof(float) * VAS_N_BINS * VAS_N_FRAMES);
  float logits[VAS_N_CLASSES], probs[VAS_N_CLASSES];

  for (int i = 0; i < count; i++) {
    if (fread(samples, sizeof(float), VAS_N_SAMPLES, fi) != (size_t)VAS_N_SAMPLES) {
      fprintf(stderr, "short read at signal %d\n", i);
      return 2;
    }
    vas_logmel(samples, feat);
    vas_nn_forward(feat, logits);
    vas_nn_softmax(logits, probs);

    fwrite(feat, sizeof(float), VAS_N_BINS * VAS_N_FRAMES, fo);
    fwrite(logits, sizeof(float), VAS_N_CLASSES, fo);
    fwrite(probs, sizeof(float), VAS_N_CLASSES, fo);
  }

  free(samples); free(feat);
  fclose(fi); fclose(fo);
  return 0;
}
