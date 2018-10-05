#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>

#include "utils/Timer.h"

#include "cnn/cnn.h"
#include "Train.h"
#include "Infer.h"

int main() {
  Timer t;

  /* Training phase */
  std::cout << "Reading test cases." << std::endl;
  std::vector<case_t> cases = readTrainingData();

  std::vector<layer_t *> layers;

  // The following layers make up the CNN in sequence.
  std::cout << "Creating layers." << std::endl;
  auto layer1 = new conv_layer_t(1, 5, 8, cases[0].data.size);
  auto layer2 = new relu_layer_t(layer1->out.size);
  auto layer3 = new pool_layer_t(2, 2, layer2->out.size);
  auto layer4 = new fc_layer_t(layer3->out.size, 10);

  // Push the layers into a vector of layers.
  layers.push_back(layer1);
  layers.push_back(layer2);
  layers.push_back(layer3);
  layers.push_back(layer4);

  // Train the CNN
  t.start();
  auto total_error = trainCNN(layers, cases);
  t.stop();
  std::cout << "Training completed.\n";
  std::cout << "  Time (baseline): " << t.seconds() << " s.\n";
  std::cout << "  Total error: " << total_error << std::endl;

  /* Test inference */
  // Load the test image
  auto image_tensor = loadImageAsTensor("../test.ppm");

  // Infer the CNN with the result image
  forward(layers, image_tensor);

  printResult(layers);

  delete layer1;
  delete layer2;
  delete layer3;
  delete layer4;

  return 0;

}
