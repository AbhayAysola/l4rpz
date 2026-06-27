#include "decompress.hpp"
#include "lz4.hpp"
#include "lz4_frame.hpp"

// decompresses one lz4 block from src (src_len bytes) into dest (which holds
// dest_capacity bytes). returns the number of uncompressed bytes, or -1 on a
// malformed block.
static __device__ int64_t parse_data_block(const uint8_t *src, size_t src_len,
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

// one thread per block — safe because we require block_independence
__global__ void decompress_kernel(const uint8_t *compressed,
                                  const LZ4BlockInfo *blocks, int num_blocks,
                                  uint8_t *output, int64_t *output_sizes,
                                  size_t max_block_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;

  uint8_t *dest = output + (size_t)i * max_block_size;
  output_sizes[i] = parse_data_block(compressed + blocks[i].data_offset,
                                     blocks[i].data_size,
                                     blocks[i].is_compressed,
                                     dest, max_block_size);
}

std::vector<uint8_t> decompress_frame(const uint8_t *data, size_t size) {
  // smallest compressed file size TODO: need to verify
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

  int num_blocks = static_cast<int>(blocks.size());
  size_t max_block_size = config.max_block_size_bytes;

  // upload compressed data and block descriptors
  uint8_t *d_compressed = nullptr;
  LZ4BlockInfo *d_blocks = nullptr;
  uint8_t *d_output = nullptr;
  int64_t *d_output_sizes = nullptr;

  cudaMalloc(&d_compressed, size);
  cudaMemcpy(d_compressed, data, size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_blocks, num_blocks * sizeof(LZ4BlockInfo));
  cudaMemcpy(d_blocks, blocks.data(), num_blocks * sizeof(LZ4BlockInfo),
             cudaMemcpyHostToDevice);

  // stride layout: block i writes to d_output + i * max_block_size
  cudaMalloc(&d_output, num_blocks * max_block_size);
  cudaMalloc(&d_output_sizes, num_blocks * sizeof(int64_t));

  int threads = 256;
  int grid = (num_blocks + threads - 1) / threads;
  decompress_kernel<<<grid, threads>>>(d_compressed, d_blocks, num_blocks,
                                      d_output, d_output_sizes, max_block_size);
  cudaDeviceSynchronize();

  // copy results back
  std::vector<int64_t> output_sizes(num_blocks);
  cudaMemcpy(output_sizes.data(), d_output_sizes,
             num_blocks * sizeof(int64_t), cudaMemcpyDeviceToHost);

  std::vector<uint8_t> h_output(num_blocks * max_block_size);
  cudaMemcpy(h_output.data(), d_output, num_blocks * max_block_size,
             cudaMemcpyDeviceToHost);

  cudaFree(d_compressed);
  cudaFree(d_blocks);
  cudaFree(d_output);
  cudaFree(d_output_sizes);

  // pack strided output into a contiguous result
  std::vector<uint8_t> result;
  for (int i = 0; i < num_blocks; i++) {
    if (output_sizes[i] < 0) return {};
    size_t off = (size_t)i * max_block_size;
    result.insert(result.end(),
                  h_output.begin() + off,
                  h_output.begin() + off + output_sizes[i]);
  }

  return result;
}
