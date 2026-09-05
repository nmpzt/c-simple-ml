# Very simple machine learning library in C

A lightweight, header-only multilayer perceptron library written in C.

## Features

- **Activation functions**:
  - Linear (`ACT_linear`)
  - ReLU (`ACT_relu`)
  - Leaky ReLU (`ACT_leaky_relu`)
  - Logistic / Sigmoid (`ACT_logistic`)
  - Tanh (`ACT_tanh`)
- **Weight Initializations**:
  - Zero Initialization (`INIT_zero`)
  - Xavier Initialization (`INIT_Xavier`)
  - He Initialization (`INIT_He`)
- **Loss Functions**:
  - Mean Squared Error (`LOSS_MSE`)
  - Absolute Squared Error (`LOSS_ASE`)
  - Binary Cross-Entropy (`LOSS_binary_cross_entropy`)
- **Save & load**: Save and load trained model weights, biases, and topologies to/from disk using clean binary serialization (`MLP_save` / `MLP_load`).

## API Reference

### Topology Management
- `void Topology_init(Topology* topo, uint16_t layers, uint16_t* neurons)`: Initializes the network structure and caches weight/value offsets.
- `void Topology_free(Topology* topo)`: Frees topology memory.

### Network Lifecycle
- `void MLP_initialise(MLP* input, Topology topo, Activation hidden_act, Activation output_act)`: Allocates memory for weights, biases, values, and gradient deltas.
- `void MLP_populate(MLP* input, Initialisation initialisation_method, unsigned int seed)`: Populates weights using Zero, Xavier, or He initialization.
- `void MLP_free(MLP* input)`: Safely frees all memory associated with the network and its topology.

### Execution & Training
- `void MLP_set_inputs(MLP* mlp, float values[])`: Loads input features into the input layer.
- `void MLP_evaluate(MLP* mlp)`: Performs forward propagation.
- `void MLP_get_outputs(MLP* mlp, float* rop)`: Extracts predictions from the output layer.
- `void MLP_backpropagate(MLP* mlp, const float* targets, Loss_func loss_func)`: Computes gradients via backpropagation for a single training sample.
- `void MLP_update_weights(MLP* mlp, float learning_rate)`: Applies gradient descent to update weights and biases.
- `void MLP_train_step(MLP* mlp, float inputs[], float targets[], float learning_rate, Loss_func loss_func)`: Runs a single forward pass, backpropagation, and weight update.
- `void MLP_train(...)`: Standard batch training loop across multiple epochs.
- `void MLP_train_logged(...)`: Training loop with periodic loss evaluation and logging callback support.

### Saving & Loading
- `int MLP_save(const MLP* mlp, const char* filename)`: Serializes network metadata, layer sizes, weights, and biases to binary.
- `int MLP_load(MLP* mlp, const char* filename)`: Restores a network from a binary model file.

## Compilation

Compile it however you want! (with `-lm`).

## Demos
There are found in the `demos` folder (shocking, I know).
- `demo0.c`: Train an XOR model and store it in the `models` folder.
- `demo1.c`: Load the XOR model from the `models` folder and test it in real time.
- `demo2.c`: Train a simple spiral seperation model.
- `plot_demo2.c`: Plots the output model of `demo2.c`. This will output a file named `output.ppm` that you'll need a .ppm viewer to view.

## Issues
- Lack of errors/fallback.
- Doesn't work with openmp for some reason: I tried to parallelise it using `omp.h`, then it just didn't work anymore.
- Doesn't output to a more universal format, just unreadable binary files: I really don't know of a universal MLP format.
- Because of the last reason, it's kinda hard to test it.