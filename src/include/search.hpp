#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// GPU context. owns all CUDA allocations and streams
// create once per process and reuse across search_file calls to avoid
// repeated cudaMalloc/cudaFree overhead between files
struct GpuContext;
GpuContext *make_gpu_context();
void        free_gpu_context(GpuContext *ctx);

// returns a pinned host buffer of at least `needed` bytes, growing if necessary
// the buffer is owned by ctx and reused across calls
uint8_t *gpu_file_buf(GpuContext *ctx, size_t needed);

// searches one lz4 frame starting at data[0]. returns byte offsets of all
// matches in the decompressed content, or nullopt on error.
// compressed_consumed: set to the number of compressed bytes the frame occupied.
// decompressed_size_out: set to the total decompressed size of the frame.
std::optional<std::vector<size_t>> search_frame(const uint8_t *data, size_t size,
                                                const std::string &pattern,
                                                GpuContext *ctx,
                                                bool case_insensitive = false,
                                                size_t *compressed_consumed = nullptr,
                                                size_t *decompressed_size_out = nullptr,
                                                bool bench = false);

// searches all lz4 frames in data (a complete file). returns byte offsets of
// all matches across all frames in decompressed order, or nullopt on error.
std::optional<std::vector<size_t>> search_file(const uint8_t *data, size_t size,
                                               const std::string &pattern,
                                               GpuContext *ctx,
                                               bool case_insensitive = false,
                                               bool bench = false);
