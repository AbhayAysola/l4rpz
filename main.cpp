#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

constexpr uint32_t FRAME_MAGIC_NUMBER = 0x184D2204;

// reads sizeof(T) bytes from file into value. returns false on read failure.
// assumes a little-endian host (x86/CUDA).
template <typename T> bool read_raw(std::ifstream &file, T &value) {
  return static_cast<bool>(
      file.read(reinterpret_cast<char *>(&value), sizeof(value)));
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
  std::streampos file_offset;
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

// decompresses one lz4 block into dest (which holds dest_capacity bytes).
// returns the number of uncompressed bytes, or -1 on a malformed
// block. little endian is assumed
int64_t parse_data_block(std::ifstream &file, LZ4BlockInfo block_info,
                         uint8_t *dest, size_t dest_capacity) {
  file.seekg(block_info.file_offset);

  if (!block_info.is_compressed) {
    if (block_info.data_size > dest_capacity) {
      return -1;
    }
    file.read(reinterpret_cast<char *>(dest), block_info.data_size);
    if (!file) {
      return -1;
    }
    return block_info.data_size;
  }

  // compressed block
  // iterate over sequences until exit condition is met
  size_t dest_pos = 0;
  while ((file.tellg() - block_info.file_offset) < block_info.data_size) {
    uint8_t token;
    if (!read_raw(file, token)) {
      return -1;
    }

    size_t num_literals = token >> 4; // high bits
    if (num_literals == 15) {
      uint8_t next_byte;
      do {
        if (!read_raw(file, next_byte)) {
          return -1;
        }
        num_literals += next_byte;
      } while (next_byte == 255);
    }

    if (num_literals > 0) {
      if (dest_pos + num_literals > dest_capacity) {
        return -1;
      }
      file.read(reinterpret_cast<char *>(dest + dest_pos), num_literals);
      if (!file) {
        return -1;
      }
      dest_pos += num_literals;
    }

    // the last sequence in a block is literals only and ends here
    // TODO: check for end of block requirements
    if ((file.tellg() - block_info.file_offset) >= block_info.data_size) {
      break;
    }

    uint16_t offset;
    if (!read_raw(file, offset)) {
      return -1;
    }

    // offset 0 is invalid; a match must not reach before the start of output
    if (offset == 0 || offset > dest_pos) {
      return -1;
    }
    size_t match_length = token & 0b00001111; // low bits
    if (match_length == 15) {
      uint8_t next_byte = 0;
      do {
        if (!read_raw(file, next_byte)) {
          return -1;
        }
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

  LZ4FrameHeaderRaw raw;
  read_raw(file, raw.magic_number);

  if (raw.magic_number != FRAME_MAGIC_NUMBER) {
    std::cout << "file corrupted!\n";
    return 1;
  }

  read_raw(file, raw.flg);
  read_raw(file, raw.bd);

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
    file.seekg(8, std::ios::cur);
  }
  if (config.dict_id_present) {
    file.seekg(4, std::ios::cur);
  }

  // TODO: verify the header_checksum
  uint8_t header_checksum = 0;
  read_raw(file, header_checksum);

  std::cout << "header parsed. max block size: " << config.max_block_size_bytes
            << " bytes" << std::endl;

  // read all the block headers
  std::vector<LZ4BlockInfo> blocks;
  while (true) {
    uint32_t block_size_field = 0;
    read_raw(file, block_size_field);

    if (!file || block_size_field == 0) { // 0x00000000 endmark
      break;
    }

    LZ4BlockInfo block;
    block.is_compressed = !(block_size_field & (1U << 31));
    block.data_size = block_size_field & ~(1U << 31);
    block.file_offset = file.tellg();

    blocks.push_back(block);

    // seek to next block header
    file.seekg(block.data_size, std::ios::cur);
    // TODO: verify block checksum
    if (config.block_checksum) {
      file.seekg(4, std::ios::cur);
    }
  }

  std::cout << "parsed " << blocks.size() << " data block headers" << std::endl;

  std::vector<uint8_t> out(config.max_block_size_bytes);

  for (size_t i = 0; i < blocks.size(); ++i) {
    int64_t uncompressed_size =
        parse_data_block(file, blocks[i], out.data(), out.size());
    if (uncompressed_size < 0) {
      std::cout << "block " << i << ": malformed/corrupt, decode failed\n";
      continue;
    }
    std::cout << "block " << i << ": size = " << blocks[i].data_size
              << " bytes, compressed = "
              << (blocks[i].is_compressed ? "yes" : "no")
              << ", offset = " << blocks[i].file_offset
              << ", uncompressed size = " << uncompressed_size << " bytes:\n";
    std::cout.write(reinterpret_cast<char *>(out.data()), uncompressed_size);
    std::cout << '\n';
  }

  return 0;
}
