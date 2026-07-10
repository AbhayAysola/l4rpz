#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "search.hpp"


static std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return {};

  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();
  if (file_size < 0) return {};
  file.seekg(0, std::ios::beg);

  // read the whole compressed file into memory once. this is also the buffer
  // we would cudaMemcpy to the device in a single transfer. size it up front
  // and do a single bulk read instead of growing byte-by-byte.
  std::vector<uint8_t> data(static_cast<size_t>(file_size));
  if (!file.read(reinterpret_cast<char *>(data.data()), file_size)) return {};
  return data;
}

// run as l4rpz [-i] <pattern> <file|dir> [file|dir ...]
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: l4rpz [-i] <pattern> <file|dir> [file|dir ...]\n";
    return 1;
  }

  bool case_insensitive = (std::string(argv[1]) == "-i");
  int pattern_arg = case_insensitive ? 2 : 1;

  if (argc < pattern_arg + 2) {
    std::cerr << "usage: l4rpz [-i] <pattern> <file|dir> [file|dir ...]\n";
    return 1;
  }

  std::string pattern(argv[pattern_arg]);

  // collect all .lz4 files, expanding any directory arguments
  std::vector<std::string> files;
  for (int i = pattern_arg + 1; i < argc; i++) {
    std::filesystem::path p(argv[i]);
    if (std::filesystem::is_directory(p)) {
      for (auto &entry : std::filesystem::recursive_directory_iterator(p)) {
        if (entry.is_regular_file() && entry.path().extension() == ".lz4")
          files.push_back(entry.path().string());
      }
    } else {
      files.push_back(argv[i]);
    }
  }

  bool multiple_files = files.size() > 1;
  bool any_error = false;

  for (const auto &filepath : files) {
    auto data = read_file(filepath);
    if (data.empty()) {
      std::cerr << filepath << ": cannot read file\n";
      any_error = true;
      continue;
    }

    auto result = search_file(data.data(), data.size(), pattern, case_insensitive);
    if (!result) {
      std::cerr << filepath << ": search failed\n";
      any_error = true;
      continue;
    }

    for (size_t offset : *result) {
      if (multiple_files) std::cout << filepath << ':';
      std::cout << offset << '\n';
    }
  }

  return any_error ? 1 : 0;
}
