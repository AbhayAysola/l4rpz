#include "decompress.hpp"
#include "lz4.hpp"
#include "lz4_frame.hpp"

// reference cpu implementation

// decompresses one lz4 block from src (src_len bytes) into dest (which holds
// dest_capacity bytes). returns the number of uncompressed bytes, or -1 on a
// malformed block.
static int64_t parse_data_block(const uint8_t *src, size_t src_len,
                            bool is_compressed, uint8_t *dest,
                            size_t dest_capacity) {
  if (!is_compressed) {
    if (src_len > dest_capacity) {
      return -1;
    }
    for (size_t i = 0; i < src_len; i++) {
      dest[i] = src[i];
    }
    return static_cast<int64_t>(src_len);
  }

  // compressed block
  // iterate over sequences until exit condition is met
  size_t src_pos = 0;
  size_t dest_pos = 0;
  while (src_pos < src_len) {
    uint8_t token = src[src_pos++];

    size_t num_literals = token >> 4; // high bits
    if (num_literals == 15) {
      uint8_t next_byte;
      do {
        if (src_pos >= src_len) {
          return -1;
        }
        next_byte = src[src_pos++];
        num_literals += next_byte;
      } while (next_byte == 255);
    }

    if (num_literals > 0) {
      if (src_pos + num_literals > src_len ||
          dest_pos + num_literals > dest_capacity) {
        return -1;
      }
      for (size_t i = 0; i < num_literals; i++) {
        dest[dest_pos + i] = src[src_pos + i];
      }
      src_pos += num_literals;
      dest_pos += num_literals;
    }

    // the last sequence in a block is literals only and ends here
    // TODO: check for end of block requirements
    if (src_pos >= src_len) {
      break;
    }

    // 2-byte little-endian match offset
    if (src_pos + 2 > src_len) {
      return -1;
    }
    uint16_t offset = load_u16_le(src + src_pos);
    src_pos += 2;

    // offset 0 is invalid; a match must not reach before the start of output
    if (offset == 0 || offset > dest_pos) {
      return -1;
    }
    size_t match_length = token & 0b00001111; // low bits
    if (match_length == 15) {
      uint8_t next_byte = 0;
      do {
        if (src_pos >= src_len) {
          return -1;
        }
        next_byte = src[src_pos++];
        match_length += next_byte;
      } while (next_byte == 255);
    }
    match_length += 4;

    if (dest_pos + match_length > dest_capacity) {
      return -1;
    }
    size_t seq_end = match_length + dest_pos;
    for (; dest_pos < seq_end; dest_pos++) {
      dest[dest_pos] = dest[dest_pos - offset];
    }
  }
  return static_cast<int64_t>(dest_pos);
}

std::vector<uint8_t> decompress_frame(const uint8_t *data, size_t size) {
  if (size < 7) return {};

  size_t pos = 0;
  LZ4FrameHeaderRaw raw;
  raw.magic_number = load_u32_le(data + pos);
  pos += sizeof(raw.magic_number);

  if (raw.magic_number != FRAME_MAGIC_NUMBER) return {};

  raw.flg = data[pos++];
  raw.bd = data[pos++];

  LZ4FrameConfig config = parse_frame_header(raw);

  if (config.version != 1) return {};
  if (!config.block_independence) return {};
  if (config.max_block_size_bytes == 0) return {};

  // TODO: deal with content_size and dict_id properly
  if (config.content_size_present) pos += 8;
  if (config.dict_id_present) pos += 4;

  // TODO: verify the header_checksum
  pos += 1; // header checksum (HC)

  // read all the block headers
  std::vector<LZ4BlockInfo> blocks;
  while (pos + 4 <= size) {
    uint32_t block_size_field = load_u32_le(data + pos);
    pos += 4;

    if (block_size_field == 0) { // 0x00000000 endmark
      break;
    }

    LZ4BlockInfo block;
    block.is_compressed = !(block_size_field & (1U << 31));
    block.data_size = block_size_field & ~(1U << 31);
    block.data_offset = pos;

    // make sure the block body is actually present in the buffer
    if (pos + block.data_size > size) break;

    blocks.push_back(block);

    // seek past the block body to the next block header
    pos += block.data_size;
    // TODO: verify block checksum
    if (config.block_checksum) pos += 4;
  }

  std::vector<uint8_t> result;
  std::vector<uint8_t> block_buf(config.max_block_size_bytes);

  for (const auto &block : blocks) {
    int64_t n = parse_data_block(data + block.data_offset, block.data_size,
                                 block.is_compressed, block_buf.data(),
                                 block_buf.size());
    if (n < 0) return {};
    result.insert(result.end(), block_buf.begin(), block_buf.begin() + n);
  }

  return result;
}
