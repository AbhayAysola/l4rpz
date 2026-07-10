#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// searches one lz4 frame starting at data[0]. returns byte offsets of all
// matches in the decompressed content, or nullopt on error.
// compressed_consumed: set to the number of compressed bytes the frame occupied.
// decompressed_size_out: set to the total decompressed size of the frame.
std::optional<std::vector<size_t>> search_frame(const uint8_t *data, size_t size,
                                                const std::string &pattern,
                                                size_t *compressed_consumed = nullptr,
                                                size_t *decompressed_size_out = nullptr);

// searches all lz4 frames in data (a complete file). returns byte offsets of
// all matches across all frames in decompressed order, or nullopt on error.
std::optional<std::vector<size_t>> search_file(const uint8_t *data, size_t size,
                                               const std::string &pattern);
