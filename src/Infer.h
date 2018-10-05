#pragma once

#include <vector>
#include <string>

#include "cnn/layer_t.h"
#include "cnn/tensor_t.h"

/**
 * Infer the CNN to detect what digit is in some image
 * @param layers    A vector of CNN layers
 * @param data      The image
 */
void forward(std::vector<layer_t *> &layers, tensor_t<float> &data);

/// @brief Load and convert an image to a tensor
tensor_t<float> loadImageAsTensor(const std::string &file_name);

/// @brief Print the output of the last layer of a CNN
void printResult(std::vector<layer_t *> layers);
