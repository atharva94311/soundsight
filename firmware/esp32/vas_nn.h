#pragma once

#include <stddef.h>
#include <stdint.h>

#include "model_weights.h"

// Largest activation tensor in the graph, in floats. Verified at boot by
// vas_nn_check_arena(); if the model is regenerated wider than this, that call
// reports it rather than letting the network scribble past the buffer.
#ifndef VAS_NN_MAX_ACT
#define VAS_NN_MAX_ACT 20000
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** feat: VAS_N_BINS * VAS_N_FRAMES standardised floats -> VAS_N_CLASSES logits. */
void vas_nn_forward(const float *feat, float *logits);

/** Softmax with the calibration temperature baked in. */
void vas_nn_softmax(const float *logits, float *probs);

int vas_nn_argmax(const float *probs);

/**
 * Walks the layer table and returns the largest activation size in floats.
 * Call once at boot and compare against VAS_NN_MAX_ACT before running anything.
 */
size_t vas_nn_required_arena(void);

#ifdef __cplusplus
}
#endif
