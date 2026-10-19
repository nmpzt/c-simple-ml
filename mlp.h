#pragma once

#include <stddef.h> // size_t
#include <stdlib.h> // malloc & free
#include <stdint.h> // for smaller numbers and bigger numbers
#include <stdio.h>
#include <math.h>   // math.h

// DECLARATIONS SECTION

// XOSHIRO256** implementation
typedef struct {
  uint64_t s[4];
} XS256_state;

static inline uint64_t rotl_u64(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline uint64_t XS256_splitmix64(uint64_t *sm_state) {
  uint64_t z = (*sm_state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

// seeds the state with the seed
void XS256_seed(XS256_state *s, uint64_t seed);

// returns from 0 to UINT64_MAX
uint64_t XS256_rand(XS256_state *s);

// MLPE stands for MLP error
typedef enum {
  MLPE_SUCCESS = 0,
  MLPE_NULL_PTR,     // null pointer passed into function
  MLPE_MALLOC,
  MLPE_FILE_ERROR,
  MLPE_WHAT,         // unknown i guess
} MLPE_CODE;

#ifndef PI_F
#define PI_F 3.14159265358979323846f
#endif

#ifndef PI2_F
#define PI2_F 6.28318548202514648438f
#endif

typedef enum {
  ACT_linear,
  ACT_relu,
  ACT_leaky_relu,
  ACT_logistic,
  ACT_tanh,
} Activation;

typedef enum {
  INIT_zero,
  INIT_Xavier,
  INIT_He,
} Initialisation;

typedef enum {
  LOSS_MSE,
  LOSS_ASE,
  LOSS_binary_cross_entropy,
} Loss_func;

// ACTivation functions
#ifndef LEAKY_RELU_CONST
#define LEAKY_RELU_CONST 0.01f
#endif

static inline float ACT_linearf(float input) {
  return input;
}

static inline float ACT_reluf(float input) {
  return (input < 0.f) ? 0.f : input;
}

static inline float ACT_leaky_reluf(float input) {
  return input * ((input < 0.f) ? LEAKY_RELU_CONST : 1.f);
}

static inline float ACT_logisticf(float input) {
  return 1.f / (1.f + expf(-input));
}

static inline float ACT_tanhf(float input) {
  return tanhf(input);
}

// and their derivatives
static inline float derv_linearf(float input) {
  return 1.f;
}

static inline float derv_reluf(float input) {
  return (input < 0.f) ? 0.f : 1.f;
}

static inline float derv_leaky_reluf(float input) {
  return (input < 0.f) ? LEAKY_RELU_CONST : 1.f;
}

static inline float derv_logisticf(float activated_value) {
  return activated_value * (1.f - activated_value);
}

static inline float derv_tanhf(float activated_value) {
  return 1.f - activated_value * activated_value;
}

// topology
typedef struct {
  uint16_t  num_layers;
  uint16_t* layer_sizes;
  size_t*   weight_offsets; // cached start index of weights for each layer
  size_t*   val_offsets;    // cached start index of biases/values for each layer
} Topology;

MLPE_CODE Topology_init(Topology* topo, uint16_t layers, uint16_t* neurons);
MLPE_CODE Topology_free(Topology* topo);

// neural network
typedef struct {
  float*  weights;
  float*  biases;
  float*  values;    // ACTivated values a
  float*  raw_sums;    // z for backprop

  float*  values_deltas;
  float*  weights_deltas;
  float*  biases_deltas;

  float*  memory_pool; // all the data live here

  Topology topo;

  Activation hidden_layer_activation;
  Activation output_layer_activation;
} MLP;

MLPE_CODE MLP_initialise(MLP* input, Topology topo, Activation hidden_act, Activation output_act);
MLPE_CODE MLP_populate(MLP* input, Initialisation initialisation_method, XS256_state* randstate);
void MLP_free(MLP* input);
MLPE_CODE MLP_set_inputs(MLP* mlp, float values[]);
MLPE_CODE MLP_get_outputs(MLP* mlp, float* dest);
MLPE_CODE MLP_evaluate(MLP* mlp);
MLPE_CODE MLP_backpropagate(MLP* mlp, const float* targets, Loss_func loss_func);
MLPE_CODE MLP_update_weights(MLP* mlp, float learning_rate);
MLPE_CODE MLP_train_step(MLP* mlp, float inputs[], float targets[], float learning_rate, Loss_func loss_func);
MLPE_CODE MLP_train(MLP* mlp,
              float** dataset_inputs,
              float** dataset_targets,
              size_t num_samples,
              size_t epochs,
              float learning_rate,
              Loss_func loss_func);
MLPE_CODE MLP_train_logged(MLP* mlp,
              float** dataset_inputs,
              float** dataset_targets,
              size_t num_samples,
              size_t epochs,
              float learning_rate,
              Loss_func loss_func,
              size_t log_step,
              void (*logging_func)(MLP* nn, size_t epoch_num, float avg_loss));
MLPE_CODE MLP_save(const MLP* mlp, const char* filename);
MLPE_CODE MLP_load(MLP* mlp, const char* filename);


// DEFINITIONS SECTION

#ifdef MLP_H_IMPLEMENTATION

// seeds the state with the seed
void XS256_seed(XS256_state *s, uint64_t seed) {
  uint64_t sm_state = seed;
  s->s[0] = XS256_splitmix64(&sm_state);
  s->s[1] = XS256_splitmix64(&sm_state);
  s->s[2] = XS256_splitmix64(&sm_state);
  s->s[3] = XS256_splitmix64(&sm_state);

  if ((s->s[0] | s->s[1] | s->s[2] | s->s[3]) == 0) {
    s->s[0] = 0x1;
  }
}

// returns from 0 to UINT64_MAX
uint64_t XS256_rand(XS256_state *s) {
  const uint64_t result = rotl_u64(s->s[1] * 5, 7) * 9; // scrambler step
  const uint64_t t = s->s[1] << 17;

  s->s[2] ^= s->s[0];
  s->s[3] ^= s->s[1];
  s->s[1] ^= s->s[2];
  s->s[0] ^= s->s[3];

  s->s[2] ^= t;
  s->s[3] = rotl_u64(s->s[3], 45);

  return result;
}

static float random_normal(XS256_state* s) {
  static const float factor = 1.f / ((float)UINT64_MAX + 1.f);
  float u1, u2;
  do {
    u1 = (float)XS256_rand(s) * factor;
  } while (u1 <= 0.0f || u1 >= 1.0f);

  u2 = (float)XS256_rand(s) * factor;

  return sqrtf(-2.f * logf(u1)) * cosf(PI2_F * u2);
}

MLPE_CODE Topology_init(Topology* topo, uint16_t layers, uint16_t* neurons) {
  if (!topo || !neurons) return MLPE_NULL_PTR;

  topo->num_layers = layers;
  topo->layer_sizes = NULL;
  topo->weight_offsets = NULL;
  topo->val_offsets = NULL;
  
  topo->layer_sizes = (uint16_t*) malloc(layers * sizeof(uint16_t));
  if (!topo->layer_sizes) goto cleanup;

  topo->weight_offsets = (size_t*) malloc(layers * sizeof(size_t));
  if (!topo->weight_offsets) goto cleanup;

  topo->val_offsets = (size_t*) malloc(layers * sizeof(size_t));
  if (!topo->val_offsets) goto cleanup;

  size_t w_acc = 0;
  size_t v_acc = 0;

  for (uint16_t i = 0; i < layers; ++i) {
    topo->layer_sizes[i] = neurons[i];

    topo->weight_offsets[i] = w_acc;
    if (i < layers - 1) {
      w_acc += (size_t)neurons[i] * neurons[i + 1];
    }

    topo->val_offsets[i] = v_acc;
    v_acc += neurons[i];
  }

  return MLPE_SUCCESS;

cleanup:
  free(topo->layer_sizes);
  free(topo->weight_offsets);
  free(topo->val_offsets);
  
  topo->num_layers = 0;
  topo->layer_sizes = NULL;
  topo->weight_offsets = NULL;
  topo->val_offsets = NULL;
  
  return MLPE_MALLOC;
}

MLPE_CODE Topology_free(Topology* topo) {
  if (!topo) return MLPE_NULL_PTR;

  topo->num_layers = 0;
  free(topo->layer_sizes);
  topo->layer_sizes = NULL;
  free(topo->weight_offsets);
  topo->weight_offsets = NULL;
  free(topo->val_offsets);
  topo->val_offsets = NULL;

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_initialise(MLP* input, Topology topo, Activation hidden_act, Activation output_act) {
  if (!input) return MLPE_NULL_PTR;
  input->topo = topo;
  input->hidden_layer_activation = hidden_act;
  input->output_layer_activation = output_act;

  size_t weights_size = 0;
  size_t all_size = 0;

  for (uint16_t i = 0; i < topo.num_layers - 1; ++i) {
    weights_size += (size_t)topo.layer_sizes[i] * topo.layer_sizes[i + 1];
  }

  for (uint16_t i = 0; i < topo.num_layers; ++i) {
    all_size += topo.layer_sizes[i];
  }

  size_t total_floats = (2 * weights_size) + (5 * all_size);
  input->memory_pool = (float*) malloc(total_floats * sizeof(float));
  if (!input->memory_pool) return MLPE_MALLOC;

  input->weights        = input->memory_pool;
  input->weights_deltas = input->weights        + weights_size;
  input->biases         = input->weights_deltas + weights_size;
  input->values         = input->biases         + all_size;
  input->raw_sums       = input->values         + all_size;
  input->values_deltas  = input->raw_sums       + all_size;
  input->biases_deltas  = input->values_deltas  + all_size;

  return MLPE_SUCCESS;
}

static inline size_t weights_index(size_t layer, size_t target, size_t source, const Topology* topo) {
  return topo->weight_offsets[layer] + (source * topo->layer_sizes[layer + 1] + target);
}

static inline size_t val_index(size_t layer, size_t ind, const Topology* topo) {
  return topo->val_offsets[layer] + ind;
}

MLPE_CODE MLP_populate(MLP* input, Initialisation initialisation_method, XS256_state* randstate) {
  if (!input) return MLPE_NULL_PTR;

  Topology* topo = &(input->topo);

  size_t all_size = 0;
  for (uint16_t i = 0; i < topo->num_layers; ++i) {
    all_size += topo->layer_sizes[i];
  }
  for (size_t i = 0; i < all_size; ++i) {
    input->biases[i] = 0.0f;
  }

  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo->num_layers - 1; ++i) {
    weights_size += (size_t)topo->layer_sizes[i] * topo->layer_sizes[i + 1];
  }

  if (initialisation_method == INIT_zero) {
    for (size_t i = 0; i < weights_size; ++i) {
      input->weights[i] = 0.0f;
    }
  } 
  else if (initialisation_method == INIT_Xavier) {
    for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
      float n_in  = (float)topo->layer_sizes[l];
      float n_out = (float)topo->layer_sizes[l + 1];
      float stddev = sqrtf(2.f / (n_in + n_out));

      for (size_t target = 0; target < topo->layer_sizes[l + 1]; ++target) {
        for (size_t source = 0; source < topo->layer_sizes[l]; ++source) {
          size_t idx = weights_index(l, target, source, topo);
          input->weights[idx] = random_normal(randstate) * stddev;
        }
      }
    }
  } 
  else if (initialisation_method == INIT_He) {
    for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
      float n_in = (float)topo->layer_sizes[l];
      float stddev = sqrtf(2.f / n_in);

      for (size_t target = 0; target < topo->layer_sizes[l + 1]; ++target) {
        for (size_t source = 0; source < topo->layer_sizes[l]; ++source) {
          size_t idx = weights_index(l, target, source, topo);
          input->weights[idx] = random_normal(randstate) * stddev;
        }
      }
    }
  }

  return MLPE_SUCCESS;
}

void MLP_free(MLP* input) {
  if (!input) return;
  free(input->memory_pool);
  input->memory_pool = NULL;
  input->weights = NULL;
  input->weights_deltas = NULL;
  input->biases = NULL;
  input->values = NULL;
  input->raw_sums = NULL;
  input->values_deltas = NULL;
  input->biases_deltas = NULL;

  Topology_free( &(input->topo) );
}

MLPE_CODE MLP_set_inputs(MLP* mlp, float values[]) {
  if (!mlp) return MLPE_NULL_PTR;
  uint16_t firstlayer_size = mlp->topo.layer_sizes[0];
  for (uint16_t i = 0; i < firstlayer_size; ++i) {
    mlp->values[val_index(0, i,  &(mlp->topo) )] = values[i];
  }

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_get_outputs(MLP* mlp, float* dest) {
  if (!mlp) return MLPE_NULL_PTR;
  const size_t last_layer_index = mlp->topo.num_layers - 1;
  const size_t last_layer_size = mlp->topo.layer_sizes[last_layer_index];

  for (size_t i = 0; i < last_layer_size; ++i) {
    dest[i] = mlp->values[val_index(last_layer_index, i, &(mlp->topo) )];
  }

  return MLPE_SUCCESS;
}

static float apply_activation(Activation act, float input) {
  switch (act) {
    case ACT_linear:       return ACT_linearf(input);
    case ACT_relu:         return ACT_reluf(input);
    case ACT_leaky_relu:   return ACT_leaky_reluf(input);
    case ACT_logistic:     return ACT_logisticf(input);
    case ACT_tanh:         return ACT_tanhf(input);
    default:               return input;
  }
}

MLPE_CODE MLP_evaluate(MLP* mlp) {
  if (!mlp) return MLPE_NULL_PTR;
  const Topology* topo = &(mlp->topo);

  for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
    uint16_t current_size = topo->layer_sizes[l];
    uint16_t next_size = topo->layer_sizes[l + 1];
    Activation act = (l == topo->num_layers - 2) ? mlp->output_layer_activation 
                                                 : mlp->hidden_layer_activation;

    for (size_t j = 0; j < next_size; ++j) {
      size_t next_v_idx = val_index(l + 1, j, topo);
      mlp->raw_sums[next_v_idx] = mlp->biases[next_v_idx];
      
      for (size_t i = 0; i < current_size; ++i) {
          size_t curr_v_idx = val_index(l, i, topo);
          size_t w_idx = weights_index(l, j, i, topo);
          
          mlp->raw_sums[next_v_idx] += mlp->weights[w_idx] * mlp->values[curr_v_idx];
      }
      
      mlp->values[next_v_idx] = apply_activation(act, mlp->raw_sums[next_v_idx]);
    }
  }

  return MLPE_SUCCESS;
}

static float apply_activation_derivative(Activation act, float activated_val) {
  switch (act) {
    case ACT_linear:       return derv_linearf(activated_val);
    case ACT_relu:         return derv_reluf(activated_val);
    case ACT_leaky_relu:   return derv_leaky_reluf(activated_val);
    case ACT_logistic:     return derv_logisticf(activated_val);
    case ACT_tanh:         return derv_tanhf(activated_val);
    default:               return 1.f;
  }
}

MLPE_CODE MLP_backpropagate(MLP* mlp, const float* targets, Loss_func loss_func) {
  if (!mlp || !targets) return MLPE_NULL_PTR;
  const Topology* topo = &(mlp->topo);
  uint16_t L = topo->num_layers - 1;
  size_t output_size = topo->layer_sizes[L];

  for (size_t j = 0; j < output_size; ++j) {
    size_t v_idx = val_index(L, j, topo);
    float a = mlp->values[v_idx];
    float t = targets[j];
    float derivative = apply_activation_derivative(mlp->output_layer_activation, a);

    float loss_derv = 0.0f;
    if (loss_func == LOSS_MSE || loss_func == LOSS_ASE) {
      loss_derv = (a - t);
    } else if (loss_func == LOSS_binary_cross_entropy) {
      float eps = 1e-15f;
      float clamped_a = fmaxf(eps, fminf(1.f - eps, a));
      loss_derv = (clamped_a - t) / (clamped_a * (1.f - clamped_a));
    }

    mlp->values_deltas[v_idx] = loss_derv * derivative;
  }

  for (int16_t l = (int16_t)L - 1; l >= 0; --l) {
    uint16_t curr_size = topo->layer_sizes[l];
    uint16_t next_size = topo->layer_sizes[l + 1];

    for (size_t i = 0; i < curr_size; ++i) {
      size_t curr_v_idx = val_index(l, i, topo);
      float error_sum = 0.0f;

      for (size_t j = 0; j < next_size; ++j) {
        size_t next_v_idx = val_index(l + 1, j, topo);
        size_t w_idx = weights_index(l, j, i, topo);
        error_sum += mlp->weights[w_idx] * mlp->values_deltas[next_v_idx];
      }

      float a = mlp->values[curr_v_idx];
      Activation act = mlp->hidden_layer_activation;
      
      mlp->values_deltas[curr_v_idx] = error_sum * apply_activation_derivative(act, a);
    }
  }

  for (uint16_t l = 0; l < topo->num_layers - 1; ++l) {
    uint16_t curr_size = topo->layer_sizes[l];
    uint16_t next_size = topo->layer_sizes[l + 1];

    for (size_t j = 0; j < next_size; ++j) {
      size_t next_v_idx = val_index(l + 1, j, topo);
      float next_delta = mlp->values_deltas[next_v_idx];

      mlp->biases_deltas[next_v_idx] = next_delta;

      for (size_t i = 0; i < curr_size; ++i) {
        size_t curr_v_idx = val_index(l, i, topo);
        size_t w_idx = weights_index(l, j, i, topo);
        
        mlp->weights_deltas[w_idx] = next_delta * mlp->values[curr_v_idx];
      }
    }
  }

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_update_weights(MLP* mlp, float learning_rate) {
  if (!mlp) return MLPE_NULL_PTR;
  const Topology* topo = &(mlp->topo);

  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo->num_layers - 1; ++i) {
    weights_size += (size_t)topo->layer_sizes[i] * topo->layer_sizes[i + 1];
  }

  for (size_t i = 0; i < weights_size; ++i) {
    mlp->weights[i] -= learning_rate * mlp->weights_deltas[i];
  }

  size_t all_size = 0;
  for (uint16_t i = 0; i < topo->num_layers; ++i) {
    all_size += topo->layer_sizes[i];
  }

  for (size_t i = 0; i < all_size; ++i) {
    mlp->biases[i] -= learning_rate * mlp->biases_deltas[i];
  }

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_train_step(MLP* mlp, float inputs[], float targets[], float learning_rate, Loss_func loss_func) {
  if (!mlp || !inputs || !targets) return MLPE_NULL_PTR;
  MLPE_CODE ecode = MLP_set_inputs(mlp, inputs);
  if (ecode) return ecode;

  ecode = MLP_evaluate(mlp);
  if (ecode) return ecode;

  ecode = MLP_backpropagate(mlp, targets, loss_func);
  if (ecode) return ecode;

  ecode = MLP_update_weights(mlp, learning_rate);
  if (ecode) return ecode;

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_train(MLP* mlp,
              float** dataset_inputs,
              float** dataset_targets,
              size_t num_samples,
              size_t epochs,
              float learning_rate,
              Loss_func loss_func) {
  if (!mlp || !dataset_inputs || !dataset_targets || !*dataset_inputs || !*dataset_targets) return MLPE_NULL_PTR;

  MLPE_CODE ecode = MLPE_SUCCESS;
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < num_samples; ++i) {
      ecode = MLP_train_step(mlp, dataset_inputs[i], dataset_targets[i], learning_rate, loss_func);
      if (ecode) return ecode;
    }
  }

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_train_logged(MLP* mlp,
              float** dataset_inputs,
              float** dataset_targets,
              size_t num_samples,
              size_t epochs,
              float learning_rate,
              Loss_func loss_func,
              size_t log_step,
              void (*logging_func)(MLP* nn, size_t epoch_num, float avg_loss)) {
  if (!mlp || !dataset_inputs || !dataset_targets || !*dataset_inputs || !*dataset_targets || !logging_func) return MLPE_NULL_PTR;

  size_t output_size = mlp->topo.layer_sizes[mlp->topo.num_layers - 1];
  float* output = (float*)malloc(output_size * sizeof(float));
  if (!output) return MLPE_MALLOC;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    for (size_t i = 0; i < num_samples; ++i) {
      MLPE_CODE ecode = MLP_train_step(mlp, dataset_inputs[i], dataset_targets[i], learning_rate, loss_func);
      if (ecode) return ecode;
    }

    if ((epoch + 1) % log_step == 0) {
      float total_loss = 0.0f;
      for (size_t i = 0; i < num_samples; ++i) {
        MLPE_CODE ecode = MLP_set_inputs(mlp, dataset_inputs[i]);
        if (ecode) return ecode;
        ecode = MLP_evaluate(mlp);
        if (ecode) return ecode;
        ecode = MLP_get_outputs(mlp, output);
        if (ecode) return ecode;

        for (size_t j = 0; j < output_size; ++j) {
          float a = output[j];
          float t = dataset_targets[i][j];
          if (loss_func == LOSS_MSE || loss_func == LOSS_ASE) {
            float diff = a - t;
            total_loss += diff * diff;
          } else if (loss_func == LOSS_binary_cross_entropy) {
            float eps = 1e-15f;
            float clamped_a = fmaxf(eps, fminf(1.f - eps, a));
            total_loss -= (t * logf(clamped_a) + (1.f - t) * logf(1.f - clamped_a));
          }
        }
      }
      float avg_loss = total_loss / (float)num_samples;
      logging_func(mlp, epoch, avg_loss);
    }
  }
  free(output);

  return MLPE_SUCCESS;
}

MLPE_CODE MLP_save(const MLP* mlp, const char* filename) {
  if (!mlp || !filename) return MLPE_NULL_PTR;
  FILE* file = fopen(filename, "wb");
  if (!file) {
    return MLPE_FILE_ERROR;
  }

  const Topology* topo = &(mlp->topo);

  fwrite(&(topo->num_layers), sizeof(uint16_t), 1, file);
  fwrite(&(mlp->hidden_layer_activation), sizeof(Activation), 1, file);
  fwrite(&(mlp->output_layer_activation), sizeof(Activation), 1, file);

  fwrite(topo->layer_sizes, sizeof(uint16_t), topo->num_layers, file);

  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo->num_layers - 1; ++i) {
      weights_size += (size_t)topo->layer_sizes[i] * topo->layer_sizes[i + 1];
  }

  size_t all_size = 0;
  for (uint16_t i = 0; i < topo->num_layers; ++i) {
      all_size += topo->layer_sizes[i];
  }

  fwrite(mlp->weights, sizeof(float), weights_size, file);
  fwrite(mlp->biases, sizeof(float), all_size, file);

  fclose(file);
  return MLPE_SUCCESS;
}

MLPE_CODE MLP_load(MLP* mlp, const char* filename) {
  if (!mlp || !filename) return MLPE_NULL_PTR;
  FILE* file = fopen(filename, "rb");
  if (!file) {
    return MLPE_FILE_ERROR;
  }

  uint16_t num_layers = 0;
  Activation hidden_act, output_act;

  if (fread(&num_layers, sizeof(uint16_t), 1, file) != 1 ||
    fread(&hidden_act, sizeof(Activation), 1, file) != 1 ||
    fread(&output_act, sizeof(Activation), 1, file) != 1) {
    fclose(file);
    return MLPE_FILE_ERROR;
  }

  uint16_t* layer_sizes = (uint16_t*)malloc(num_layers * sizeof(uint16_t));
  if (!layer_sizes) {
    fclose(file);
    return MLPE_MALLOC;
  }
  if (fread(layer_sizes, sizeof(uint16_t), num_layers, file) != num_layers) {
    free(layer_sizes);
    fclose(file);
    return MLPE_FILE_ERROR;
  }

  Topology topo;
  MLPE_CODE ecode = Topology_init(&topo, num_layers, layer_sizes);
  if (ecode) {
    free(layer_sizes);
    fclose(file);
    return ecode;
  }
  free(layer_sizes);
  layer_sizes = NULL;

  ecode = MLP_initialise(mlp, topo, hidden_act, output_act);
  if (ecode) {
    Topology_free(&topo);
    fclose(file);
    return ecode;
  }

  size_t weights_size = 0;
  for (uint16_t i = 0; i < topo.num_layers - 1; ++i) {
    weights_size += (size_t)topo.layer_sizes[i] * topo.layer_sizes[i + 1];
  }

  size_t all_size = 0;
  for (uint16_t i = 0; i < topo.num_layers; ++i) {
    all_size += topo.layer_sizes[i];
  }

  if (fread(mlp->weights, sizeof(float), weights_size, file) != weights_size ||
    fread(mlp->biases, sizeof(float), all_size, file) != all_size) {
    MLP_free(mlp);
    fclose(file);
    return MLPE_FILE_ERROR;
  }

  fclose(file);
  return MLPE_SUCCESS;
}

#endif // MLP_H_IMPLEMENTATION
