// XOR network demo

#include <stdio.h>
#include <assert.h>

#include "../mlp.h"

int main(void) {
  Topology mytopo;
  Topology_init(&mytopo, 3, (uint16_t[]) {2, 3, 1});

  MLP mymlp;

  MLP_initialise(&mymlp, mytopo, ACT_logistic, ACT_logistic);

  MLP_populate(&mymlp, INIT_Xavier, 666);

  float* learning_inputs[] = {
    (float[]) {0.f, 0.f},
    (float[]) {1.f, 0.f},
    (float[]) {0.f, 1.f},
    (float[]) {1.f, 1.f},
  };

  float* learning_targets[] = {
    (float[]) {0.f},
    (float[]) {1.f},
    (float[]) {1.f},
    (float[]) {0.f},
  };

  MLP_train(&mymlp, (float**) learning_inputs, (float**) learning_targets, 4, 65536, 0.1f, LOSS_MSE);

  float output[1];
  for (int i = 0; i < 4; ++i) {
    MLP_set_inputs(&mymlp, learning_inputs[i]);
    MLP_evaluate(&mymlp);
    MLP_get_outputs(&mymlp, output);
    printf("{%g, %g} => %g\n", learning_inputs[i][0], learning_inputs[i][1], output[0]);
  }

  assert(!MLP_save(&mymlp, "models/xor_model.bin"));

  MLP_free(&mymlp);

  return 0;
}

