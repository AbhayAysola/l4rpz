#include <cstdint>
#include <filesystem>
#include <fstream>
#include <getopt.h>
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

static void usage() {
  std::cerr << "usage: l4rpz [-i] [-c] [-L] [-q] <pattern> <file|dir> [file|dir ...]\n";
}

// run as l4rpz [options] <pattern> <file|dir> [file|dir ...]
// default output: prints filenames that have matches (one per file)
int main(int argc, char *argv[]) {
  static const struct option long_opts[] = {
      {"ignore-case",           no_argument, nullptr, 'i'},
      {"count",                 no_argument, nullptr, 'c'},
      {"files-without-matches", no_argument, nullptr, 'L'},
      {"quiet",                 no_argument, nullptr, 'q'},
      {nullptr, 0, nullptr, 0},
  };

  bool case_insensitive      = false;
  bool count                 = false;
  bool files_without_matches = false;
  bool quiet                 = false;

  int opt;
  while ((opt = getopt_long(argc, argv, "icLq", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'i': case_insensitive      = true; break;
      case 'c': count                 = true; break;
      case 'L': files_without_matches = true; break;
      case 'q': quiet                 = true; break;
      default: usage(); return 1;
    }
  }

  if (optind + 2 > argc) { usage(); return 1; }

  std::string pattern(argv[optind]);

  std::vector<std::string> files;
  for (int i = optind + 1; i < argc; i++) {
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

  bool any_match = false;
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

    bool has_match = !result->empty();
    if (has_match) any_match = true;

    if (quiet && has_match) return 0;

    if (files_without_matches) {
      if (!has_match) std::cout << filepath << '\n';
    } else if (count) {
      if (has_match) std::cout << filepath << ':' << result->size() << '\n';
    } else {
      if (has_match) std::cout << filepath << '\n';
    }
  }

  if (quiet) return any_match ? 0 : 1;
  return any_error ? 1 : 0;
}
