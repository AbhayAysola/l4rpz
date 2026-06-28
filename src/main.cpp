#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "search.hpp"

// run as l4rpz <pattern> <file.lz4>
int main(int argc, char *argv[]) {
  if (argc < 3) {
    return 1;
  }

  std::string pattern(argv[1]);

  std::ifstream file(argv[2], std::ios::binary);
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

  auto result = search_frame(data.data(), data.size(), pattern);
  if (!result) {
    std::cerr << "search failed!\n";
    return 1;
  }

  for (size_t offset : *result) {
    std::cout << offset << '\n';
  }

  return 0;
}
