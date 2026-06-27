#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// little-endian byte loads — explicit, so they don't depend on host byte order
inline HD uint16_t load_u16_le(const uint8_t *p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

inline uint32_t load_u32_le(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

struct LZ4FrameHeaderRaw {
  uint32_t magic_number;
  uint8_t flg;
  uint8_t bd;
};

struct LZ4FrameConfig {
  uint32_t version;
  bool block_independence;
  bool block_checksum;
  bool content_size_present;
  bool content_checksum;
  bool dict_id_present;
  uint32_t max_block_size_bytes;
};

struct LZ4BlockInfo {
  uint32_t data_size;
  bool is_compressed;
  size_t data_offset; // byte offset of the block body within the file buffer
};