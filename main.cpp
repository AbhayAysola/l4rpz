#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

const uint32_t FRAME_MAGIC_NUMBER = 0x184D2204;

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

int parse_data_block(std::ifstream &file, LZ4BlockInfo block_info,
                     uint8_t *dest) {
  file.seekg(block_info.file_offset);

  if (!block_info.is_compressed) {
    file.read(reinterpret_cast<char *>(dest), block_info.data_size);
    return 0;
  }

  size_t dest_pos = 0;
  // compressed block
  // iterate over sequences until exit condition is met
  while ((file.tellg() - block_info.file_offset) < block_info.data_size) {
    uint8_t token;
    file.read(reinterpret_cast<char *>(&token), 1);

    size_t num_literals = token >> 4; // high bits
    if (num_literals == 15) {
      uint8_t next_byte;
      do {
        file.read(reinterpret_cast<char *>(&next_byte), 1);
        num_literals += next_byte;
      } while (next_byte == 255);
    }

    if (num_literals > 0) {
      file.read(reinterpret_cast<char *>(dest + dest_pos), num_literals);
      dest_pos += num_literals;
    }
    if ((file.tellg() - block_info.file_offset) >= block_info.data_size) {
      break;
    }

    uint16_t offset;
    file.read(reinterpret_cast<char *>(&offset), 2);

    if (offset == 0) {
      return 1;
    }
    size_t match_length = token & 0b00001111; // low bits
    if (match_length == 15) {
      uint8_t next_byte = 0;
      do {
        file.read(reinterpret_cast<char *>(&next_byte), 1);
        match_length += next_byte;
      } while (next_byte == 255);
    }
    match_length += 4;

    size_t seq_end = match_length + dest_pos;
    for (; dest_pos < seq_end; dest_pos++) {
      dest[dest_pos] = dest[dest_pos - offset]; 
    }
  }
  return 0;
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
  file.read(reinterpret_cast<char *>(&raw.magic_number),
            sizeof(raw.magic_number));

  if (raw.magic_number != FRAME_MAGIC_NUMBER) {
    return 1;
  }

  file.read(reinterpret_cast<char *>(&raw.flg), sizeof(raw.flg));
  file.read(reinterpret_cast<char *>(&raw.bd), sizeof(raw.bd));

  LZ4FrameConfig config = parse_frame_header(raw);

  if (config.version != 1 || !config.block_independence) {
    return 1;
  }

  if (config.content_size_present) {
    file.seekg(8, std::ios::cur);
  }
  if (config.dict_id_present) {
    file.seekg(4, std::ios::cur);
  }

  uint8_t hc = 0;
  file.read(reinterpret_cast<char *>(&hc), sizeof(hc));

  std::cout << "header parsed. max block size: " << config.max_block_size_bytes
            << " bytes" << std::endl;

  std::vector<LZ4BlockInfo> blocks;
  while (true) {
    uint32_t block_size_field = 0;
    file.read(reinterpret_cast<char *>(&block_size_field),
              sizeof(block_size_field));

    if (!file || block_size_field == 0) {
      break;
    }

    LZ4BlockInfo block;
    block.is_compressed = !(block_size_field & (1U << 31));
    block.data_size = block_size_field & ~(1U << 31);
    block.file_offset = file.tellg();

    blocks.push_back(block);

    file.seekg(block.data_size, std::ios::cur);

    if (config.block_checksum) {
      file.seekg(4, std::ios::cur);
    }
  }

  std::cout << "parsed " << blocks.size() << " data block headers" << std::endl;

  uint8_t *out = (uint8_t *)malloc(config.max_block_size_bytes);
  if (out == NULL) {
    return 1;
  }

  for (size_t i = 0; i < blocks.size(); ++i) {
    std::cout << "block " << i << ": size = " << blocks[i].data_size
              << " bytes, compressed = "
              << (blocks[i].is_compressed ? "yes" : "no")
              << ", offset = " << blocks[i].file_offset << std::endl;
    parse_data_block(file, blocks[i], out);
    std::cout << out << '\n';
  }

  file.close();
  return 0;
}
