#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

constexpr uint32_t FRAME_MAGIC_NUMBER = 0x184D2204;

// little-endian byte loads — explicit, so they don't depend on host byte order
inline uint16_t load_u16_le(const uint8_t *p) {
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

// decompresses one lz4 block from src (src_len bytes) into dest (which holds
// dest_capacity bytes). returns the number of uncompressed bytes, or -1 on a
// malformed block. 
int64_t parse_data_block(const uint8_t *src, size_t src_len,
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    return 1;
  }

  std::ifstream file(argv[1], std::ios::binary);
  if (!file.is_open()) {
    return 1;
  }

  // read the whole compressed file into memory once. this is also the buffer
  // we would cudaMemcpy to the device in a single transfer. size it up front
  // and do a single bulk read instead of growing byte-by-byte.
  file.seekg(0, std::ios::end);
  std::streamoff file_size = file.tellg();
  if (file_size < 0) {
    std::cout << "could not determine file size!\n";
    return 1;
  }
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(static_cast<size_t>(file_size));
  if (!file.read(reinterpret_cast<char *>(data.data()), file_size)) {
    std::cout << "could not read file!\n";
    return 1;
  }

  // smallest possible frame: 4-byte magic + flg + bd + 1-byte header checksum
  // TODO: not sure if this is necessary
  if (data.size() < 7) {
    std::cout << "file too small!\n";
    return 1;
  }

  size_t pos = 0;
  LZ4FrameHeaderRaw raw;
  raw.magic_number = load_u32_le(data.data() + pos);
  pos += sizeof(raw.magic_number);

  if (raw.magic_number != FRAME_MAGIC_NUMBER) {
    std::cout << "file corrupted!\n";
    return 1;
  }

  raw.flg = data[pos++];
  raw.bd = data[pos++];

  LZ4FrameConfig config = parse_frame_header(raw);

  if (config.version != 1) {
    std::cout << "incompatible version!\n";
    return 1;
  }
  if (!config.block_independence) {
    std::cout << "please compress with block independence enabled!\n";
    return 1;
  }
  if (config.max_block_size_bytes == 0) {
    std::cout << "invalid/reserved block max size!\n";
    return 1;
  }

  // TODO: deal with content_size and dict_id properly
  if (config.content_size_present) {
    pos += 8;
  }
  if (config.dict_id_present) {
    pos += 4;
  }

  // TODO: verify the header_checksum
  pos += 1; // header checksum (HC)

  std::cout << "header parsed. max block size: " << config.max_block_size_bytes
            << " bytes" << std::endl;

  // read all the block headers
  std::vector<LZ4BlockInfo> blocks;
  while (pos + 4 <= data.size()) {
    uint32_t block_size_field = load_u32_le(data.data() + pos);
    pos += 4;

    if (block_size_field == 0) { // 0x00000000 endmark
      break;
    }

    LZ4BlockInfo block;
    block.is_compressed = !(block_size_field & (1U << 31));
    block.data_size = block_size_field & ~(1U << 31);
    block.data_offset = pos;

    // make sure the block body is actually present in the buffer
    if (pos + block.data_size > data.size()) {
      std::cout << "truncated block body!\n";
      break;
    }

    blocks.push_back(block);

    // seek past the block body to the next block header
    pos += block.data_size;
    // TODO: verify block checksum
    if (config.block_checksum) {
      pos += 4;
    }
  }

  std::cout << "parsed " << blocks.size() << " data block headers" << std::endl;

  std::vector<uint8_t> out(config.max_block_size_bytes);

  for (size_t i = 0; i < blocks.size(); ++i) {
    int64_t uncompressed_size = parse_data_block(
        data.data() + blocks[i].data_offset, blocks[i].data_size,
        blocks[i].is_compressed, out.data(), out.size());
    if (uncompressed_size < 0) {
      std::cout << "block " << i << ": malformed/corrupt, decode failed\n";
      continue;
    }
    std::cout << "block " << i << ": size = " << blocks[i].data_size
              << " bytes, compressed = "
              << (blocks[i].is_compressed ? "yes" : "no")
              << ", offset = " << blocks[i].data_offset
              << ", uncompressed size = " << uncompressed_size << " bytes:\n";
    std::cout.write(reinterpret_cast<char *>(out.data()), uncompressed_size);
    std::cout << '\n';
  }

  return 0;
}
