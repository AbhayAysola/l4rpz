#include <cstdint>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>
#include "lz4_frame.hpp"
#include "search.hpp"


// reads path into ctx's pinned file buffer. returns the number of bytes read,
// or 0 on failure. pinned memory lets the driver DMA directly on cudaMemcpy
// without staging through a temporary buffer
static size_t read_file_pinned(const std::string &path, GpuContext *ctx) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) return 0;

  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();
  if (file_size <= 0) return 0;
  file.seekg(0, std::ios::beg);

  auto *buf = gpu_file_buf(ctx, static_cast<size_t>(file_size));
  if (!buf) return 0;
  if (!file.read(reinterpret_cast<char *>(buf), file_size)) return 0;
  return static_cast<size_t>(file_size);
}

static void usage() {
  std::cerr << "usage: l4rpz [-i] [-c] [-L] [-q] [-V] [-m N] <pattern> <file|dir> [file|dir ...]\n";
}

// run as l4rpz [options] <pattern> <file|dir> [file|dir ...]
int main(int argc, char *argv[]) {
  static const struct option long_opts[] = {
      {"ignore-case",           no_argument,       nullptr, 'i'}, // case-insensitive pattern matching
      {"only-count",            no_argument,       nullptr, 'c'}, // print filename:count instead of filename:offset per match
      {"files-without-matches", no_argument,       nullptr, 'L'}, // print only filenames with no matches
      {"quiet",                 no_argument,       nullptr, 'q'}, // no output; exit 0 if any match, 1 if none
      {"max-count",             required_argument, nullptr, 'm'}, // stop after N matches per file
      {"no-verify",             no_argument,       nullptr, 'V'}, // skip header and block checksum verification
      {"bench",                 no_argument,       nullptr, 'B'}, // emit per-frame timing to stderr
      {nullptr, 0, nullptr, 0},
  };

  bool case_insensitive      = false;
  bool count                 = false;
  bool files_without_matches = false;
  bool quiet                 = false;
  bool bench                 = false;
  bool verify_checksums      = true;
  int  max_count             = 0;

  int opt;
  while ((opt = getopt_long(argc, argv, "icLqm:VB", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'i': case_insensitive      = true; break;
      case 'c': count                 = true; break;
      case 'L': files_without_matches = true; break;
      case 'q': quiet                 = true; break;
      case 'V': verify_checksums      = false; break;
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

  GpuContext *ctx = make_gpu_context();
  if (!ctx) { std::cerr << "l4rpz: failed to initialize GPU\n"; return 2; }

  bool any_match = false;
  bool any_error = false;

  for (const auto &filepath : files) {
    size_t file_size = read_file_pinned(filepath, ctx);
    if (file_size == 0) {
      std::cerr << filepath << ": cannot read file\n";
      any_error = true;
      continue;
    }

    const uint8_t *data = gpu_file_buf(ctx, file_size);

    // allow non .lz4 files also
    if (file_size < 4 || load_u32_le(data) != FRAME_MAGIC_NUMBER) {
      std::cerr << filepath << ": not an LZ4 file\n";
      any_error = true;
      continue;
    }

    auto result = search_file(data, file_size, pattern, ctx, case_insensitive, bench, verify_checksums);
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
      for (size_t offset : *result)
        std::cout << filepath << ':' << offset << '\n';
    }
  }

  free_gpu_context(ctx);
  if (any_error) return 2;
  return any_match ? 0 : 1;
}
