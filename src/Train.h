#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <byteswap.h>
#include <memory>
#include <iomanip>
#include <iostream>
#include "cnn/tensor_t.h"
#include "cnn/layer_t.h"
#include "cnn/cnn.h"

// Some constants:
constexpr size_t img_dim = 28;      // All training images in the MNIST data set are 28x28.
constexpr size_t num_digits = 10;

// case including input data and label
struct case_t {
  tensor_t<float> data;
  tensor_t<float> out;
};

/// @brief Read the MNIST training data from
std::vector<case_t> readTrainingData();

/**
 * @brief Train the CNN with one image
 * @param layers    The layers of the CNN
 * @param train_tensor      The training data
 * @param expected_output  The expected outcome (labels)
 * @return          The error
 */
float trainIterate(std::vector<layer_t *> &layers, tensor_t<float> &train_tensor, tensor_t<float> &expected_output);

/**
 * @brief Train the CNN with a whole dataset of images
 * @param layers    Layers to train
 * @param cases     Labeled test cases for training
 * @return          The total error after training the whole data set
 */
float trainCNN(std::vector<layer_t *> &layers, std::vector<case_t> &cases);