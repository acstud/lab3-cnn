#pragma once

#include "types.h"
#include "tensor_t.h"

// the type of layers
#pragma pack(push, 1)

struct layer_t {
  layer_type type;
  tensor_t<float> grads_in;
  tensor_t<float> in;
  tensor_t<float> out;

  layer_t(layer_type type, const tensor_t<float> &grads_in, const tensor_t<float> &in, const tensor_t<float> &out)
      : type(type), grads_in(grads_in), in(in), out(out) {}
};

#pragma pack(pop)
