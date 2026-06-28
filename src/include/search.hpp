#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// searches a compressed lz4 frame for pattern. returns byte offsets of all
// matches in the decompressed content, or nullopt on error.
std::optional<std::vector<size_t>> search_frame(const uint8_t *data, size_t size,
                                                const std::string &pattern);
