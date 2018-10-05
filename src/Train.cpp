#include "Train.h"
#include "utils/io.h"

std::vector<case_t> readTrainingData() {
  std::vector<case_t> cases{};

  auto train_image = readFile("../train-images.idx3-ubyte");
  auto train_labels = readFile("../train-labels.idx1-ubyte");

  // Swap endianness of case count:
  uint32_t case_count = bswap_32(*(uint32_t *) (train_image.data() + sizeof(uint32_t)));

  // Convert each image to a test case
  for (int i = 0; i < case_count; i++) {
    case_t c{tensor_t<float>(img_dim, img_dim, 1), tensor_t<float>(num_digits, 1, 1)};

    // Actual images start at offset 16
    uint8_t *img = train_image.data() + 16 + i * (img_dim * img_dim);

    // Actual labels start at offset 8
    uint8_t *label = train_labels.data() + 8 + i;

    // Normalize the pixel intensity values to a floating point number between 0 and 1.
    for (int x = 0; x < img_dim; x++) {
      for (int y = 0; y < img_dim; y++) {
        c.data(x, y, 0) = img[x + y * img_dim] / 255.f;
      }
    }

    // Convert the labels to a floating point number at the correct digit index
    for (int b = 0; b < num_digits; b++) {
      c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;
    }

    cases.push_back(c);
  }

  return cases;
}

float trainIterate(std::vector<layer_t *> &layers, tensor_t<float> &train_tensor, tensor_t<float> &expected_output) {
  // Activate each layer with the training tensor as input
  for (int i = 0; i < layers.size(); i++) {
    if (i == 0)
      activate(layers[i], train_tensor);
    else
      activate(layers[i], layers[i - 1]->out);
  }

  // Check the difference with the label / expected output
  tensor_t<float> grads = layers.back()->out - expected_output;

  // Calculate the gradients
  for (ssize_t i = layers.size() - 1; i >= 0; i--) {
    if (i == layers.size() - 1)
      calc_grads(layers[i], grads);
    else
      calc_grads(layers[i], layers[i + 1]->grads_in);
  }

  // Potentially update the weights
  for (auto &layer : layers) {
    fix_weights(layer);
  }

  // Calculate the error
  float err = 0;
  for (int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++) {
    float f = expected_output.data[i];
    if (f > 0.5)
      err += abs(grads.data[i]);
  }
  return err * 100;
}

float trainCNN(std::vector<layer_t *> &layers, std::vector<case_t> &cases) {
  float total_error = 0.0f;
  // Iterate over all the test cases
  int count = 0;
  for (case_t &c : cases) {
    // Train the model with a new test case, and retreive the new error.
    float xerr = trainIterate(layers, c.data, c.out);
    // Accumulate the total error for all test cases.
    total_error += xerr;

    // Once every 6000 training iterations, report the total error.
    count++;
    if (count % 6000 == 0) {
      std::cout << "Case " << std::setw(5) << count << ". Err=" << total_error / count << std::endl;
    }
  }
  return total_error;
}
