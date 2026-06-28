#include "search.hpp"
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

// one thread per block: decompress then search. storing strided offsets
// (i * max_block_size + o) so the cpu can convert to logical offsets using
// prefix sums of output_sizes after the kernel completes.
__global__ void decompress_and_search_kernel(
    const uint8_t *compressed, const LZ4BlockInfo *blocks, int num_blocks,
    uint8_t *decompressed, int64_t *output_sizes, size_t max_block_size,
    const uint8_t *pattern, int pattern_len,
    size_t *matches, int *match_count, int max_matches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;

  uint8_t *dest = decompressed + (size_t)i * max_block_size;
  int64_t n = parse_data_block(compressed + blocks[i].data_offset,
                               blocks[i].data_size, blocks[i].is_compressed,
                               dest, max_block_size);
  output_sizes[i] = n;
  if (n < 0 || pattern_len == 0) return;

  for (int64_t o = 0; o <= n - pattern_len; o++) {
    bool found = true;
    for (int j = 0; j < pattern_len && found; j++) {
      if (dest[o + j] != pattern[j]) found = false;
    }
    if (found) {
      int idx = atomicAdd(match_count, 1);
      if (idx < max_matches)
        matches[idx] = (size_t)i * max_block_size + (size_t)o;
    }
  }
}

std::optional<std::vector<size_t>> search_frame(const uint8_t *data, size_t size, const std::string &pattern) {
  // smallest compressed file size TODO: need to verify
  if (size < 7) return std::nullopt;

  size_t pos = 0;
  LZ4FrameHeaderRaw raw;
  raw.magic_number = load_u32_le(data + pos);
  pos += sizeof(raw.magic_number);

  if (raw.magic_number != FRAME_MAGIC_NUMBER) return std::nullopt;

  raw.flg = data[pos++];
  raw.bd = data[pos++];

  LZ4FrameConfig config = parse_frame_header(raw);

  if (config.version != 1) return std::nullopt;
  if (!config.block_independence) return std::nullopt;
  if (config.max_block_size_bytes == 0) return std::nullopt;

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

  // upper bound on matches: every position could be a match
  size_t max_matches = (size_t)num_blocks * max_block_size / pattern.size() + 1;
  int pattern_len = static_cast<int>(pattern.size());

  // upload compressed data, block descriptors, and pattern
  uint8_t *d_compressed = nullptr;
  LZ4BlockInfo *d_blocks = nullptr;
  uint8_t *d_decompressed = nullptr;
  int64_t *d_output_sizes = nullptr;
  uint8_t *d_pattern = nullptr;
  size_t *d_matches = nullptr;
  int *d_match_count = nullptr;

  cudaMalloc(&d_compressed, size);
  cudaMemcpy(d_compressed, data, size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_blocks, num_blocks * sizeof(LZ4BlockInfo));
  cudaMemcpy(d_blocks, blocks.data(), num_blocks * sizeof(LZ4BlockInfo),
             cudaMemcpyHostToDevice);

  // stride layout: block i writes to d_decompressed + i * max_block_size
  cudaMalloc(&d_decompressed, (size_t)num_blocks * max_block_size);
  cudaMalloc(&d_output_sizes, num_blocks * sizeof(int64_t));

  cudaMalloc(&d_pattern, pattern_len);
  cudaMemcpy(d_pattern, pattern.data(), pattern_len, cudaMemcpyHostToDevice);

  cudaMalloc(&d_matches, max_matches * sizeof(size_t));
  cudaMalloc(&d_match_count, sizeof(int));
  cudaMemset(d_match_count, 0, sizeof(int));

  int threads = 256;
  int grid = (num_blocks + threads - 1) / threads;
  decompress_and_search_kernel<<<grid, threads>>>(
      d_compressed, d_blocks, num_blocks, d_decompressed, d_output_sizes,
      max_block_size, d_pattern, pattern_len, d_matches, d_match_count,
      static_cast<int>(max_matches));
  cudaDeviceSynchronize();

  // copy results back
  int match_count = 0;
  cudaMemcpy(&match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost);

  std::vector<int64_t> output_sizes(num_blocks);
  cudaMemcpy(output_sizes.data(), d_output_sizes,
             num_blocks * sizeof(int64_t), cudaMemcpyDeviceToHost);

  std::vector<size_t> h_matches(match_count);
  cudaMemcpy(h_matches.data(), d_matches, match_count * sizeof(size_t),
             cudaMemcpyDeviceToHost);

  cudaFree(d_compressed);
  cudaFree(d_blocks);
  cudaFree(d_decompressed);
  cudaFree(d_output_sizes);
  cudaFree(d_pattern);
  cudaFree(d_matches);
  cudaFree(d_match_count);

  // compute prefix sums to convert strided offsets to logical offsets
  // TODO: experiment, maybe we should do this on the gpu?
  std::vector<size_t> prefix(num_blocks + 1, 0);
  for (int i = 0; i < num_blocks; i++) {
    if (output_sizes[i] < 0) return std::nullopt;
    prefix[i + 1] = prefix[i] + static_cast<size_t>(output_sizes[i]);
  }

  std::vector<size_t> result;
  result.reserve(match_count);
  for (int i = 0; i < match_count; i++) {
    int block_idx = static_cast<int>(h_matches[i] / max_block_size);
    size_t within = h_matches[i] % max_block_size;
    result.push_back(prefix[block_idx] + within);
  }

  return result;
}