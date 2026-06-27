#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// decompresses a full lz4 frame. returns the decompressed bytes, or an empty
// vector on error.
std::vector<uint8_t> decompress_frame(const uint8_t *data, size_t size);
