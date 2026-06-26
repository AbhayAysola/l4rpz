#include <cstdint>
#include "lz4.hpp"
#include "lz4_frame.hpp"


LZ4FrameConfig parse_frame_header(const LZ4FrameHeaderRaw &raw) {
  LZ4FrameConfig config;

  config.version = (raw.flg >> 6) & 3;
  config.block_independence = (raw.flg >> 5) & 1;
  config.block_checksum = (raw.flg >> 4) & 1;
  config.content_size_present = (raw.flg >> 3) & 1;
  config.content_checksum = (raw.flg >> 2) & 1;
  config.dict_id_present = raw.flg & 1;

  uint8_t size_index = (raw.bd >> 4) & 7;
  switch (size_index) {
  case 4:
    config.max_block_size_bytes = 64 * 1024;
    break;
  case 5:
    config.max_block_size_bytes = 256 * 1024;
    break;
  case 6:
    config.max_block_size_bytes = 1024 * 1024;
    break;
  case 7:
    config.max_block_size_bytes = 4 * 1024 * 1024;
    break;
  default:
    config.max_block_size_bytes = 0;
    break;
  }

  return config;
}