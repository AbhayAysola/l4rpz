#include <cstdint>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include "lz4_frame.hpp"
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
  // TODO: for larger files we'd need to stream, or else it would not fit in RAM
  std::vector<uint8_t> data(static_cast<size_t>(file_size));
  if (!file.read(reinterpret_cast<char *>(data.data()), file_size)) return {};
  return data;
}

static void usage() {
  std::cerr << "usage: l4rpz [-i] [-c] [-L] [-q] [-m N] <pattern> <file|dir> [file|dir ...]\n";
}

// run as l4rpz [options] <pattern> <file|dir> [file|dir ...]
// default output: prints filenames that have matches (one per file)
int main(int argc, char *argv[]) {
  static const struct option long_opts[] = {
      {"ignore-case",           no_argument,       nullptr, 'i'},
      {"count",                 no_argument,       nullptr, 'c'},
      {"files-without-matches", no_argument,       nullptr, 'L'},
      {"quiet",                 no_argument,       nullptr, 'q'},
      {"max-count",             required_argument, nullptr, 'm'},
      {"bench",                 no_argument,       nullptr, 'B'},
      {nullptr, 0, nullptr, 0},
  };

  bool case_insensitive      = false;
  bool count                 = false;
  bool files_without_matches = false;
  bool quiet                 = false;
  bool bench                 = false;
  int  max_count             = 0;

  int opt;
  while ((opt = getopt_long(argc, argv, "icLqm:B", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'i': case_insensitive      = true; break;
      case 'c': count                 = true; break;
      case 'L': files_without_matches = true; break;
      case 'q': quiet                 = true; break;
      case 'B': bench                 = true; break;
      case 'm':
        max_count = std::atoi(optarg);
        if (max_count <= 0) { std::cerr << "l4rpz: -m requires a positive integer\n"; return 1; }
        break;
      default: usage(); return 1;
    }
  }

  if (optind + 2 > argc) { usage(); return 1; }

  std::string pattern(argv[optind]);

  // read all the files and recursively read .lz4 in directories
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

    // allow non .lz4 files also
    if (data.size() < 4 || load_u32_le(data.data()) != FRAME_MAGIC_NUMBER) {
      std::cerr << filepath << ": not an LZ4 file\n";
      any_error = true;
      continue;
    }

    auto result = search_file(data.data(), data.size(), pattern, case_insensitive, bench);
    if (!result) {
      std::cerr << filepath << ": invalid LZ4 file\n";
      any_error = true;
      continue;
    }

    // truncate if max_count is passed
    // TODO: early exit on the gpu directly
    if (max_count > 0 && (int)result->size() > max_count)
      result->resize((size_t)max_count);

    bool has_match = !result->empty();
    if (has_match) any_match = true;

    if (quiet && has_match && !any_error) return 0;

    if (files_without_matches) {
      if (!has_match) std::cout << filepath << '\n';
    } else if (count) {
      if (has_match) std::cout << filepath << ':' << result->size() << '\n';
    } else {
      if (has_match) std::cout << filepath << '\n';
    }
  }

  if (any_error) return 2;
  return any_match ? 0 : 1;
}
