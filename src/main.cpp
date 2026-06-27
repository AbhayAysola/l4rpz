#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "decompress.hpp"

// run as l4rpz <file.lz4>
int main(int argc, char *argv[]) {
  if (argc < 2) {
    return 1;
  }

  std::ifstream file(argv[1], std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "cannot open file!\n";
    return 1;
  }

  // read the whole compressed file into memory once. this is also the buffer
  // we would cudaMemcpy to the device in a single transfer. size it up front
  // and do a single bulk read instead of growing byte-by-byte.
  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();
  if (file_size < 0) {
    std::cerr << "could not determine file size!\n";
    return 1;
  }
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(static_cast<size_t>(file_size));
  if (!file.read(reinterpret_cast<char *>(data.data()), file_size)) {
    std::cerr << "could not read file!\n";
    return 1;
  }

  std::vector<uint8_t> out = decompress_frame(data.data(), data.size());
  if (out.empty()) {
    std::cerr << "decompression failed!\n";
    return 1;
  }

  std::cout.write(reinterpret_cast<const char *>(out.data()), out.size());
  return 0;
}
