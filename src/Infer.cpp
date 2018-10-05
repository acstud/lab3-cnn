#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "utils/io.h"
#include "Infer.h"
#include "cnn/cnn.h"

void forward(std::vector<layer_t *> &layers, tensor_t<float> &data) {
  for (int i = 0; i < layers.size(); i++) {
    if (i == 0)
      activate(layers[i], data);
    else
      activate(layers[i], layers[i - 1]->out);
  }
}

tensor_t<float> loadImageAsTensor(const std::string &file_name) {
  // Read the test image for inference
  auto data = readFile(file_name);

  // If the image was loaded
  if (!data.empty()) {

    // Iterate over the image data until we find the actual pixel
    uint8_t *usable = data.data();
    while (*(uint32_t *) usable != 0x0A353532)
      usable++;

#pragma pack(push, 1)
    struct RGB {
      uint8_t r, g, b;
    };
#pragma pack(pop)

    RGB *rgb = (RGB *) usable;

    // Pre-processing for the test image
    tensor_t<float> image(28, 28, 1);
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        RGB rgb_ij = rgb[i * 28 + j];
        image(j, i, 0) = (((float) rgb_ij.r
            + rgb_ij.g
            + rgb_ij.b)
            / (3.0f * 255.f));
      }
    }
    return image;
  } else {
    throw std::runtime_error("Could not open image.");
  }
}

void printResult(std::vector<layer_t *> layers) {

  std::cout << "Number, probability:" << std::endl;
  // Show the result of the output of the last layer in the CNN network
  for (int i = 0; i < 10; i++) {
    std::cout << "  " << i << " : " << std::fixed << std::setw(2) << (layers.back()->out(i, 0, 0) * 100.0f) << "%"
              << std::endl;
  }
}
