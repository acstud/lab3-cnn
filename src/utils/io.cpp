#include "io.h"

std::vector<uint8_t> readFile(const std::string &file_name) {
  std::ifstream f(file_name, std::ios::binary | std::ios::ate);
  std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);

  auto buffer = std::vector<uint8_t>(static_cast<unsigned long>(size));
  f.read((char *) buffer.data(), size);
  return buffer;
}
