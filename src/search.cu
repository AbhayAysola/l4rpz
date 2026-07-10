#include <algorithm>
#include "search.hpp"
#include "lz4.hpp"
#include "lz4_frame.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"

// returns std::nullopt on any cuda error — used for setup/teardown calls only.
// for errors inside the main processing loop we set error=true and let the
// drain path clean up properly before returning.
#define CUDA_CHECK(expr) \
    do { if ((expr) != cudaSuccess) return std::nullopt; } while(0)

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

static __device__ uint8_t lower_char(uint8_t c) {
  return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + 32) : c;
}

// one thread per block: decompress then search. storing strided offsets
// (i * max_block_size + o) so the cpu can convert to logical offsets using
// prefix sums of output_sizes after the kernel completes.
__global__ void decompress_and_search_kernel(
    const uint8_t *compressed, const LZ4BlockInfo *blocks, int num_blocks,
    uint8_t *decompressed, int64_t *output_sizes, size_t max_block_size,
    const uint8_t *pattern, int pattern_len, bool case_insensitive,
    size_t *matches, int *match_count, int max_matches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;

  uint8_t *dest = decompressed + (size_t)i * max_block_size;
  int64_t n = parse_data_block(compressed + blocks[i].data_offset,
                               blocks[i].data_size, blocks[i].is_compressed,
                               dest, max_block_size);
  output_sizes[i] = n;
  if (n < 0 || pattern_len == 0) return;

  // naive pattern search. can be optimized maybe?
  for (int64_t o = 0; o <= n - pattern_len; o++) {
    bool found = true;
    for (int j = 0; j < pattern_len && found; j++) {
      uint8_t tc = case_insensitive ? lower_char(dest[o + j]) : dest[o + j];
      if (tc != pattern[j]) found = false;
    }
    if (found) {
      // we need atomicAdd here since multiple threads write to match_count at once
      // maybe theres a better way?
      int idx = atomicAdd(match_count, 1);
      if (idx < max_matches)
        matches[idx] = (size_t)i * max_block_size + (size_t)o;
    }
  }
}

std::optional<std::vector<size_t>> search_frame(const uint8_t *data, size_t size,
                                                 const std::string &pattern,
                                                 bool case_insensitive,
                                                 size_t *compressed_consumed,
                                                 size_t *decompressed_size_out) {
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

  // verifythe header checksum: HC = (xxHash32(frame_descriptor) >> 8) & 0xFF
  // frame descriptor spans from the FLG byte up to (not including) HC
  {
    const uint8_t *descriptor = data + 4; // starts at FLG
    size_t descriptor_len = pos - 4;
    uint8_t computed_hc = (XXH32(descriptor, descriptor_len, 0) >> 8) & 0xFF;
    if (computed_hc != data[pos]) return std::nullopt;
  }
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

    pos += block.data_size;

    // verify block checksum (xxHash32 of compressed block data)
    if (config.block_checksum) {
      if (pos + 4 > size) break;
      uint32_t stored   = load_u32_le(data + pos);
      uint32_t computed = XXH32(data + block.data_offset, block.data_size, 0);
      if (computed != stored) return std::nullopt;
      pos += 4;
    }

    blocks.push_back(block);
  }

  // skip optional 4-byte content checksum that follows the endmark
  // TODO: verify the content checksum (maybe implement xxhash on the gpu?)
  if (config.content_checksum) pos += 4;
  if (compressed_consumed) *compressed_consumed = pos;

  int num_blocks = static_cast<int>(blocks.size());
  if (num_blocks == 0) {
    if (decompressed_size_out) *decompressed_size_out = 0;
    return std::vector<size_t>{};
  }
  size_t max_block_size = config.max_block_size_bytes;
  int pattern_len = static_cast<int>(pattern.size());

  // --- fixed allocations: stay on device across all chunks ---
  uint8_t    *d_compressed = nullptr;
  LZ4BlockInfo *d_blocks   = nullptr;
  uint8_t    *d_pattern    = nullptr;

  CUDA_CHECK(cudaMalloc(&d_compressed, size));
  CUDA_CHECK(cudaMemcpy(d_compressed, data, size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_blocks, (size_t)num_blocks * sizeof(LZ4BlockInfo)));
  CUDA_CHECK(cudaMemcpy(d_blocks, blocks.data(), (size_t)num_blocks * sizeof(LZ4BlockInfo),
             cudaMemcpyHostToDevice));

  // lowercase the pattern on the host so the kernel only needs to fold text bytes
  std::string pat = pattern;
  if (case_insensitive)
    for (char &c : pat) c = (char)std::tolower((unsigned char)c);

  CUDA_CHECK(cudaMalloc(&d_pattern, (size_t)pattern_len));
  CUDA_CHECK(cudaMemcpy(d_pattern, pat.data(), (size_t)pattern_len, cudaMemcpyHostToDevice));

  // --- compute chunk size from available VRAM ---
  size_t vram_free;
  CUDA_CHECK(cudaMemGetInfo(&vram_free, nullptr));

  const size_t headroom = 256ULL << 20; // 256 MB cushion for driver + other allocations TODO: check if this is good enough
  size_t available = (vram_free > headroom) ? vram_free - headroom : 0;

  // worst-case matches per block: every position in the decompressed block
  size_t max_matches_per_block = max_block_size / (size_t)std::max(pattern_len, 1) + 1;
  size_t per_block_bytes = max_block_size                           // d_decompressed
                         + sizeof(int64_t)                          // d_output_sizes
                         + max_matches_per_block * sizeof(size_t);  // d_matches

  // two buffer slots, so each slot gets half the available space
  size_t blocks_per_chunk = (available / 2) / std::max(per_block_bytes, (size_t)1);
  blocks_per_chunk = std::clamp(blocks_per_chunk, (size_t)1, (size_t)num_blocks);

  size_t max_matches_per_chunk = blocks_per_chunk * max_matches_per_block;

  // --- double-buffered device + pinned host allocations ---
  uint8_t  *d_decompressed[2] = {};
  int64_t  *d_output_sizes[2] = {};
  size_t   *d_matches[2]      = {};
  int      *d_match_count[2]  = {};

  int64_t  *h_output_sizes[2] = {};
  size_t   *h_matches[2]      = {};
  int      *h_match_count[2]  = {};

  for (int b = 0; b < 2; b++) {
    CUDA_CHECK(cudaMalloc(&d_decompressed[b], blocks_per_chunk * max_block_size));
    CUDA_CHECK(cudaMalloc(&d_output_sizes[b], blocks_per_chunk * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_matches[b],  max_matches_per_chunk * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_match_count[b],  sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_output_sizes[b], blocks_per_chunk * sizeof(int64_t)));
    CUDA_CHECK(cudaMallocHost(&h_matches[b],      max_matches_per_chunk * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&h_match_count[b],  sizeof(int)));
    *h_match_count[b] = 0;
  }

  cudaStream_t streams[2];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));

  // prefix[i] = logical byte offset at the start of block i across the whole frame
  std::vector<size_t> prefix(num_blocks + 1, 0);
  std::vector<size_t> result;

  struct PendingChunk { int chunk_idx, block_start, block_count; };
  PendingChunk pending[2] = {{-1, 0, 0}, {-1, 0, 0}};
  bool error = false;

  // sync slot s, fill prefix sums, convert strided offsets to logical, clear pending
  auto flush_slot = [&](int s) -> bool {
    if (pending[s].chunk_idx < 0) return true;
    if (cudaStreamSynchronize(streams[s]) != cudaSuccess) return false;
    int bstart = pending[s].block_start;
    int bcount = pending[s].block_count;
    for (int i = 0; i < bcount; i++) {
      if (h_output_sizes[s][i] < 0) return false;
      prefix[bstart + i + 1] = prefix[bstart + i] + (size_t)h_output_sizes[s][i];
    }
    int mc = *h_match_count[s];
    for (int i = 0; i < mc; i++) {
      int blk = (int)(h_matches[s][i] / max_block_size);
      size_t off = h_matches[s][i] % max_block_size;
      result.push_back(prefix[bstart + blk] + off);
    }
    pending[s].chunk_idx = -1;
    return true;
  };

  int num_chunks = (num_blocks + (int)blocks_per_chunk - 1) / (int)blocks_per_chunk;

  for (int ci = 0; ci < num_chunks && !error; ci++) {
    int s = ci % 2;

    // flush this slot — waits for chunk ci-2's kernel + D2H and processes results
    if (!flush_slot(s)) { error = true; break; }

    int bstart = ci * (int)blocks_per_chunk;
    int bcount = (int)std::min((size_t)(num_blocks - bstart), blocks_per_chunk);

    int threads = 256;
    int grid    = (bcount + threads - 1) / threads;

    if (cudaMemsetAsync(d_match_count[s], 0, sizeof(int), streams[s]) != cudaSuccess) {
      error = true; break;
    }

    decompress_and_search_kernel<<<grid, threads, 0, streams[s]>>>(
        d_compressed, d_blocks + bstart, bcount,
        d_decompressed[s], d_output_sizes[s], max_block_size,
        d_pattern, pattern_len, case_insensitive,
        d_matches[s], d_match_count[s], (int)max_matches_per_chunk);

    if (cudaGetLastError() != cudaSuccess) { error = true; break; }

    // async D2H — all on the same stream so they're ordered after the kernel
    if (cudaMemcpyAsync(h_output_sizes[s], d_output_sizes[s],
                    (size_t)bcount * sizeof(int64_t), cudaMemcpyDeviceToHost, streams[s]) != cudaSuccess ||
        cudaMemcpyAsync(h_match_count[s], d_match_count[s],
                    sizeof(int), cudaMemcpyDeviceToHost, streams[s]) != cudaSuccess ||
        cudaMemcpyAsync(h_matches[s], d_matches[s],
                    max_matches_per_chunk * sizeof(size_t), cudaMemcpyDeviceToHost, streams[s]) != cudaSuccess) {
      error = true; break;
    }

    pending[s] = {ci, bstart, bcount};
  }

  // drain remaining slots — flush earlier chunk first so prefix sums are in order
  if (!error) {
    int first = 0, second = 1;
    if (pending[0].chunk_idx >= 0 && pending[1].chunk_idx >= 0 &&
        pending[1].chunk_idx < pending[0].chunk_idx) {
      first = 1; second = 0;
    }
    if (!flush_slot(first) || !flush_slot(second)) error = true;
  } else {
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);

  for (int b = 0; b < 2; b++) {
    cudaFree(d_decompressed[b]);
    cudaFree(d_output_sizes[b]);
    cudaFree(d_matches[b]);
    cudaFree(d_match_count[b]);
    cudaFreeHost(h_output_sizes[b]);
    cudaFreeHost(h_matches[b]);
    cudaFreeHost(h_match_count[b]);
  }

  cudaFree(d_compressed);
  cudaFree(d_blocks);
  cudaFree(d_pattern);

  if (error) return std::nullopt;
  if (decompressed_size_out) *decompressed_size_out = prefix[num_blocks];
  return result;
}

std::optional<std::vector<size_t>> search_file(const uint8_t *data, size_t size,
                                                const std::string &pattern,
                                                bool case_insensitive) {
  std::vector<size_t> result;
  size_t pos = 0;
  size_t decompressed_base = 0;

  while (pos + 4 <= size) {
    // stop at anything that isn't an LZ4 frame magic number
    if (load_u32_le(data + pos) != FRAME_MAGIC_NUMBER) break;

    size_t consumed = 0, decomp_size = 0;
    auto frame_matches = search_frame(data + pos, size - pos, pattern,
                                      case_insensitive, &consumed, &decomp_size);
    if (!frame_matches) return std::nullopt;

    for (size_t off : *frame_matches)
      result.push_back(decompressed_base + off);

    decompressed_base += decomp_size;
    if (consumed == 0) break; // shouldn't happen, but guard against infinite loop
    pos += consumed;
  }

  return result;
}