#pragma once

#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "gradient_t.h"

#pragma pack(push, 1)

// this is a fully connected layer
struct fc_layer_t : layer_t {
  //layer_type type = layer_type::fc;
  // backward path
  //tensor_t<float> grads_in;
  //tensor_t<float> in;
  //tensor_t<float> out;
  std::vector<float> input;
  tensor_t<float> weights;
  // backward path
  std::vector<gradient_t> gradients;

  // construction of fully connected layer
  fc_layer_t(tdsize in_size, int out_size) :
      layer_t(layer_type::fc,
              tensor_t<float>(in_size.x, in_size.y, in_size.z),
              tensor_t<float>(in_size.x, in_size.y, in_size.z),
              tensor_t<float>(out_size, 1, 1)), weights(in_size.x * in_size.y * in_size.z, out_size, 1) {
    input = std::vector<float>(static_cast<unsigned long>(out_size));
    gradients = std::vector<gradient_t>(static_cast<unsigned long>(out_size));

    int maxval = in_size.x * in_size.y * in_size.z;

    // weight initialization
    for (int i = 0; i < out_size; i++)
      for (int h = 0; h < in_size.x * in_size.y * in_size.z; h++)
        weights(h, i, 0) = 2.19722f / maxval * rand() / float(RAND_MAX);
    // 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
  }

  // this is activate function, sigmoid function
  float activator_function(float x) {
    //return tanhf( x );
    float sig = 1.0f / (1.0f + exp(-x));
    return sig;
  }

  // this is the derivative of sigmoid function
  float activator_derivative(float x) {
    //float t = tanhf( x );
    //return 1 - t * t;
    float sig = 1.0f / (1.0f + exp(-x));
    return sig * (1 - sig);
  }

  void activate(tensor_t<float> &in) {
    this->in = in;
    activate();
  }

  int map(point_t d) {
    return d.z * (in.size.x * in.size.y) +
        d.y * (in.size.x) +
        d.x;
  }

  // this is the forward path
  void activate() {
    for (int n = 0; n < out.size.x; n++) {
      float inputv = 0;

      for (int i = 0; i < in.size.x; i++)
        for (int j = 0; j < in.size.y; j++)
          for (int z = 0; z < in.size.z; z++) {
            int m = map({i, j, z});
            inputv += in(i, j, z) * weights(m, n, 0);
          }

      input[n] = inputv;

      // activate function
      out(n, 0, 0) = activator_function(inputv);
    }
  }

  // this is weight update
  void fix_weights() {
    for (int n = 0; n < out.size.x; n++) {
      gradient_t &grad = gradients[n];
      for (int i = 0; i < in.size.x; i++)
        for (int j = 0; j < in.size.y; j++)
          for (int z = 0; z < in.size.z; z++) {
            int m = map({i, j, z});
            float &w = weights(m, n, 0);
            w = update_weight(w, grad, in(i, j, z));
          }

      update_gradient(grad);
    }
  }

  // this is the gradients calculation
  void calc_grads(tensor_t<float> &grad_next_layer) {
    grads_in.clear();
    for (int n = 0; n < out.size.x; n++) {
      gradient_t &grad = gradients[n];
      grad.grad = grad_next_layer(n, 0, 0) * activator_derivative(input[n]);

      for (int i = 0; i < in.size.x; i++)
        for (int j = 0; j < in.size.y; j++)
          for (int z = 0; z < in.size.z; z++) {
            int m = map({i, j, z});
            grads_in(i, j, z) += grad.grad * weights(m, n, 0);
          }
    }
  }
};

#pragma pack(pop)
