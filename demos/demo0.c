// XOR network demo

#include <stdio.h>
#include <assert.h>

#define MLP_H_IMPLEMENTATION
#include "../mlp.h"

int main(void) {
  XS256_state rng;
  XS256_seed(&rng, 666);

  Topology mytopo;
  Topology_init(&mytopo, 3, (uint16_t[]) {2, 3, 1});

  MLP mymlp;

  MLP_initialise(&mymlp, mytopo, ACT_logistic, ACT_logistic);

  MLP_populate(&mymlp, INIT_Xavier, &rng);

  float learning_inputs[] = {
    0.f, 0.f,
    1.f, 0.f,
    0.f, 1.f,
    1.f, 1.f,
  };

  float learning_targets[] = {
    0.f,
    1.f,
    1.f,
    0.f,
  };

  MLP_train_config config = {
    .learning_rate = 0.1f,
    .loss_func = LOSS_MSE,
    .epochs = 65536,
    .log_step = 0,
    .logging_func = NULL
  };

  MLP_train(&mymlp, learning_inputs, learning_targets, 4, &config);

  float output[1];
  for (int i = 0; i < 4; ++i) {
    MLP_set_inputs(&mymlp, learning_inputs + 2*i);
    MLP_evaluate(&mymlp);
    MLP_get_outputs(&mymlp, output);
    printf("{%g, %g} => %g\n", learning_inputs[2*i], learning_inputs[1 + 2*i], output[0]);
  }

  assert(!MLP_save(&mymlp, "models/xor_model.bin"));

  MLP_free(&mymlp);

  return 0;
}
