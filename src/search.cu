#include <algorithm>
#include <chrono>
#include <cstdio>
#include "search.hpp"
#include "lz4.hpp"
#include "lz4_frame.hpp"

#define XXH_INLINE_ALL
#include "xxhash.h"

// returns std::nullopt on any cuda error — used for memcpy/memset calls inside
// search_frame. setup errors go through GpuContext::ensure_* which return bool
// do while trick from tsoding on youtube
#define CUDA_CHECK(expr) \
    do { if ((expr) != cudaSuccess) return std::nullopt; } while(0)

// grow a device allocation in place. no-op if already large enough
static bool dev_grow(void **ptr, size_t *cap, size_t needed) {
  if (needed <= *cap) return true;
  cudaFree(*ptr); *ptr = nullptr;
  if (cudaMalloc(ptr, needed) != cudaSuccess) return false;
  *cap = needed;
  return true;
}

// grow a pinned host allocation in place. no-op if already large enough
static bool pin_grow(void **ptr, size_t *cap, size_t needed) {
  if (needed <= *cap) return true;
  cudaFreeHost(*ptr); *ptr = nullptr;
  if (cudaMallocHost(ptr, needed) != cudaSuccess) return false;
  *cap = needed;
  return true;
}

struct GpuContext {
  cudaStream_t  streams[2] = {};

  // device buffers — per-file, grown lazily
  uint8_t *d_compressed = nullptr;
  LZ4BlockInfo *d_blocks = nullptr;
  uint8_t *d_pattern = nullptr;
  size_t d_compressed_cap = 0;
  size_t d_blocks_cap = 0;
  size_t d_pattern_cap = 0;

  // double-buffered slot allocations
  uint8_t *d_decompressed[2]  = {};
  int64_t *d_output_sizes[2]  = {};
  size_t *d_matches[2] = {};
  int *d_match_count[2] = {};  // always sizeof(int), allocated in make_gpu_context
  size_t d_decomp_cap = 0;     // bytes per slot
  size_t d_outsizes_cap = 0;   // bytes per slot
  size_t d_matches_cap = 0;    // bytes per slot

  // pinned host buffers
  uint8_t *h_file = nullptr;
  size_t h_file_cap = 0;
  int64_t *h_output_sizes[2] = {};
  size_t *h_matches[2] = {};
  int *h_match_count[2] = {};    // always sizeof(int), allocated in make_gpu_context

  bool ensure_compressed(size_t n) {return dev_grow((void**)&d_compressed, &d_compressed_cap, n);}
  bool ensure_blocks(size_t n) {return dev_grow((void**)&d_blocks, &d_blocks_cap, n);}
  bool ensure_pattern(size_t n) {return dev_grow((void**)&d_pattern, &d_pattern_cap, n);}
  bool ensure_file(size_t n) {return pin_grow((void**)&h_file, &h_file_cap, n);}

  bool ensure_slots(size_t decomp, size_t osizes, size_t matches) {
    if (decomp > d_decomp_cap) {
      for (int b = 0; b < 2; b++) {
        cudaFree(d_decompressed[b]);
        if (cudaMalloc(&d_decompressed[b], decomp) != cudaSuccess) return false;
      }
      d_decomp_cap = decomp;
    }
    if (osizes > d_outsizes_cap) {
      for (int b = 0; b < 2; b++) {
        cudaFree(d_output_sizes[b]);
        cudaFreeHost(h_output_sizes[b]);
        if (cudaMalloc(&d_output_sizes[b], osizes) != cudaSuccess) return false;
        if (cudaMallocHost(&h_output_sizes[b], osizes) != cudaSuccess) return false;
      }
      d_outsizes_cap = osizes;
    }
    if (matches > d_matches_cap) {
      for (int b = 0; b < 2; b++) {
        cudaFree(d_matches[b]);
        cudaFreeHost(h_matches[b]);
        if (cudaMalloc(&d_matches[b], matches) != cudaSuccess) return false;
        if (cudaMallocHost(&h_matches[b], matches) != cudaSuccess) return false;
      }
      d_matches_cap = matches;
    }
    return true;
  }
};

GpuContext *make_gpu_context() {
  auto *ctx = new GpuContext();
  if (cudaStreamCreate(&ctx->streams[0]) != cudaSuccess ||
    cudaStreamCreate(&ctx->streams[1]) != cudaSuccess ||
    cudaMalloc(&ctx->d_match_count[0], sizeof(int)) != cudaSuccess ||
    cudaMalloc(&ctx->d_match_count[1], sizeof(int)) != cudaSuccess ||
    cudaMallocHost(&ctx->h_match_count[0], sizeof(int)) != cudaSuccess ||
    cudaMallocHost(&ctx->h_match_count[1], sizeof(int)) != cudaSuccess) {
    free_gpu_context(ctx);
    return nullptr;
  }
  return ctx;
}

void free_gpu_context(GpuContext *ctx) {
  if (!ctx) return;
  cudaStreamDestroy(ctx->streams[0]);
  cudaStreamDestroy(ctx->streams[1]);
  cudaFree(ctx->d_compressed);
  cudaFree(ctx->d_blocks);
  cudaFree(ctx->d_pattern);
  for (int b = 0; b < 2; b++) {
    cudaFree(ctx->d_decompressed[b]);
    cudaFree(ctx->d_output_sizes[b]);
    cudaFree(ctx->d_matches[b]);
    cudaFree(ctx->d_match_count[b]);
    cudaFreeHost(ctx->h_output_sizes[b]);
    cudaFreeHost(ctx->h_matches[b]);
    cudaFreeHost(ctx->h_match_count[b]);
  }
  cudaFreeHost(ctx->h_file);
}

uint8_t *gpu_file_buf(GpuContext *ctx, size_t needed) {
  return ctx->ensure_file(needed) ? ctx->h_file : nullptr;
}

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
    size_t i = 0;
    while (i + 8 <= src_len) {
      uint64_t tmp;
      __builtin_memcpy(&tmp, src + i, 8);
      __builtin_memcpy(dest + i, &tmp, 8);
      i += 8;
    }
    while (i < src_len) { dest[i] = src[i]; i++; };
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
      size_t i = 0;
      while (i + 8 <= num_literals) {
        uint64_t tmp;
        __builtin_memcpy(&tmp, src + src_pos + i, 8);
        __builtin_memcpy(dest + dest_pos + i, &tmp, 8);
        i += 8;
      }
      while (i < num_literals) { dest[dest_pos + i] = src[src_pos + i]; i++; }
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
    size_t seq_end = dest_pos + match_length;
    if (offset >= 8) {
      while (dest_pos + 8 <= seq_end) {
        uint64_t tmp;
        __builtin_memcpy(&tmp, dest + dest_pos - offset, 8);
        __builtin_memcpy(dest + dest_pos, &tmp, 8);
        dest_pos += 8;
      }
    } else if (offset >= 4) {
      while (dest_pos + 4 <= seq_end) {
        uint32_t tmp;
        __builtin_memcpy(&tmp, dest + dest_pos - offset, 4);
        __builtin_memcpy(dest + dest_pos, &tmp, 4);
        dest_pos += 4;
      }
    } else if (offset >= 2) {
      while (dest_pos + 2 <= seq_end) {
        uint16_t tmp;
        __builtin_memcpy(&tmp, dest + dest_pos - offset, 2);
        __builtin_memcpy(dest + dest_pos, &tmp, 2);
        dest_pos += 2;
      }
    }
    // tail (and all of offset==1 case)
    while (dest_pos < seq_end) {
      dest[dest_pos] = dest[dest_pos - offset];
      dest_pos++;
    }
  }
  return static_cast<int64_t>(dest_pos);
}

static __device__ uint8_t lower_char(uint8_t c) {
  return (c >= 'A' && c <= 'Z') ? (uint8_t)(c + 32) : c;
}

// one cuda block per lz4 block
// thread 0 decompresses, then all threads cooperate on search
// keeping decompress and search in one kernel means the decompressed bytes are
// still in l2 cache when the search threads read them, avoiding a round-trip
// through dram between two separate kernel launches
// blockIdx.x  — which lz4 block (0..bcount-1)
// threadIdx.x — sub-range within the decompressed block (search phase only)
__global__ void decompress_and_search_kernel(
    const uint8_t *compressed, const LZ4BlockInfo *blocks, int num_blocks,
    uint8_t *decompressed, int64_t *output_sizes, size_t max_block_size,
    const uint8_t *pattern, int pattern_len, bool case_insensitive,
    size_t *matches, int *match_count, int max_matches) {
  extern __shared__ uint8_t s_pat[];

  int block_idx = blockIdx.x;
  if (block_idx >= num_blocks) return;

  // thread 0 decompresses while the other warps load the pattern into shared
  // memory.  both happen before the barrier so neither stalls waiting for the other
  if (threadIdx.x == 0) {
    uint8_t *dest = decompressed + (size_t)block_idx * max_block_size;
    output_sizes[block_idx] = parse_data_block(
        compressed + blocks[block_idx].data_offset,
        blocks[block_idx].data_size, blocks[block_idx].is_compressed,
        dest, max_block_size);
  }
  for (int i = threadIdx.x; i < pattern_len; i += blockDim.x)
    s_pat[i] = pattern[i];
  __syncthreads();

  int64_t n = output_sizes[block_idx];
  if (n < pattern_len) return;

  const uint8_t *src = decompressed + (size_t)block_idx * max_block_size;
  int64_t positions  = n - pattern_len + 1;
  int64_t per_thread = (positions + blockDim.x - 1) / blockDim.x;
  int64_t start      = (int64_t)threadIdx.x * per_thread;
  int64_t end        = min(start + per_thread, positions);

  // TODO: better search algorithm
  for (int64_t o = start; o < end; o++) {
    bool found = true;
    for (int j = 0; j < pattern_len && found; j++) {
      uint8_t tc = case_insensitive ? lower_char(src[o + j]) : src[o + j];
      if (tc != s_pat[j]) found = false;
    }
    if (found) {
      int idx = atomicAdd(match_count, 1);
      if (idx < max_matches)
        matches[idx] = (size_t)block_idx * max_block_size + (size_t)o;
    }
  }
}

// naive cpu pattern search over a small buffer. needle is already lowercased
// when case_insensitive is true (matches kernel convention). outputs to `out`
// with each match offset shifted by `base`
static void cpu_search(const uint8_t *src, size_t src_len,
                       const uint8_t *pat, int pat_len,
                       bool case_insensitive, size_t base,
                       std::vector<size_t> &out) {
  if (pat_len <= 0 || src_len < (size_t)pat_len) return;
  for (size_t i = 0; i <= src_len - (size_t)pat_len; i++) {
    bool match = true;
    for (int j = 0; j < pat_len && match; j++) {
      uint8_t c = src[i + j];
      if (case_insensitive) c = (uint8_t)std::tolower((unsigned char)c);
      match = (c == pat[j]);
    }
    if (match) out.push_back(base + i);
  }
}

std::optional<std::vector<size_t>> search_frame(const uint8_t *data, size_t size,
                                                 const std::string &pattern,
                                                 GpuContext *ctx,
                                                 bool case_insensitive,
                                                 size_t *compressed_consumed,
                                                 size_t *decompressed_size_out,
                                                 bool bench,
                                                 bool verify_checksums) {
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;
  auto t_frame_start = Clock::now();
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
  if (verify_checksums) {
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
      if (verify_checksums) {
        uint32_t stored   = load_u32_le(data + pos);
        uint32_t computed = XXH32(data + block.data_offset, block.data_size, 0);
        if (computed != stored) return std::nullopt;
      }
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

  // ensure per file device buffers are large enough, then upload
  if (!ctx->ensure_compressed(size)) return std::nullopt;
  CUDA_CHECK(cudaMemcpy(ctx->d_compressed, data, size, cudaMemcpyHostToDevice));

  if (!ctx->ensure_blocks((size_t)num_blocks * sizeof(LZ4BlockInfo))) return std::nullopt;
  CUDA_CHECK(cudaMemcpy(ctx->d_blocks, blocks.data(), (size_t)num_blocks * sizeof(LZ4BlockInfo),
             cudaMemcpyHostToDevice));

  // lowercase the pattern on the host so the kernel only needs to fold text bytes
  std::string pat = pattern;
  if (case_insensitive)
    for (char &c : pat) c = (char)std::tolower((unsigned char)c);

  if (!ctx->ensure_pattern((size_t)pattern_len)) return std::nullopt;
  CUDA_CHECK(cudaMemcpy(ctx->d_pattern, pat.data(), (size_t)pattern_len, cudaMemcpyHostToDevice));

  // --- compute chunk size from available VRAM ---
  size_t vram_free;
  CUDA_CHECK(cudaMemGetInfo(&vram_free, nullptr));

  const size_t headroom = 256ULL << 20; // 256 MB cushion for driver + other allocations TODO: check if this is good enough
  size_t available = (vram_free > headroom) ? vram_free - headroom : 0;

  // cap match buffer at 1M entries (~8MB) regardless of block/chunk size.
  // matches beyond this are silently dropped (same as -m behaviour).
  // without this cap, cudaMallocHost allocates gigabytes of pinned memory for
  // files with large block sizes and short patterns.
  static constexpr size_t MAX_MATCHES_PER_CHUNK = 1ULL << 20; // 1M matches

  size_t per_block_bytes = max_block_size    // d_decompressed
                         + sizeof(int64_t);  // d_output_sizes
  // match buffer cost is shared across the chunk, not per-block

  // two buffer slots, so each slot gets half the available space
  size_t blocks_per_chunk = (available / 2) / std::max(per_block_bytes, (size_t)1);
  blocks_per_chunk = std::clamp(blocks_per_chunk, (size_t)1, (size_t)num_blocks);

  size_t max_matches_per_chunk = MAX_MATCHES_PER_CHUNK;

  // ensure double-buffered slot allocations are large enough
  if (!ctx->ensure_slots(blocks_per_chunk * max_block_size,
                         blocks_per_chunk * sizeof(int64_t),
                         max_matches_per_chunk * sizeof(size_t))) return std::nullopt;

  // prefix[i] = logical byte offset at the start of block i across the whole frame
  std::vector<size_t> prefix(num_blocks + 1, 0);
  std::vector<size_t> result;

  // seam stitching state: tail bytes saved from the last flushed chunk's final block
  std::vector<uint8_t> seam_tail_buf;
  size_t seam_tail_base  = 0; // global decompressed offset of seam_tail_buf[0]
  int    seam_tail_block = -1; // global block index the tail came from

  struct PendingChunk { int chunk_idx, block_start, block_count; };
  PendingChunk pending[2] = {{-1, 0, 0}, {-1, 0, 0}};
  bool error = false;

  // sync slot s, fill prefix sums, convert strided offsets to logical, stitch seams, clear pending
  auto flush_slot = [&](int s) -> bool {
    if (pending[s].chunk_idx < 0) return true;
    if (cudaStreamSynchronize(ctx->streams[s]) != cudaSuccess) return false;
    int bstart = pending[s].block_start;
    int bcount = pending[s].block_count;
    for (int i = 0; i < bcount; i++) {
      if (ctx->h_output_sizes[s][i] < 0) return false;
      prefix[bstart + i + 1] = prefix[bstart + i] + (size_t)ctx->h_output_sizes[s][i];
    }
    int mc = *ctx->h_match_count[s];
    for (int i = 0; i < mc; i++) {
      int blk = (int)(ctx->h_matches[s][i] / max_block_size);
      size_t off = ctx->h_matches[s][i] % max_block_size;
      result.push_back(prefix[bstart + blk] + off);
    }

    // seam stitching: find matches that straddle block boundaries
    // these are missed by the gpu kernel which searches each block independently.
    // seam window = last (plen-1) bytes of block i + first (plen-1) bytes of block i+1.
    // any match in this window necessarily spans the boundary, so no duplicates with gpu.
    if (pattern_len > 1) {
      const uint8_t *cpat = (const uint8_t *)pat.data();

      // small synchronous D2H: stream already synced, these are tiny (at most plen-1 bytes)
      auto d2h = [&](size_t dev_off, size_t len, uint8_t *dst) -> bool {
        return cudaMemcpy(dst, ctx->d_decompressed[s] + dev_off, len,
                          cudaMemcpyDeviceToHost) == cudaSuccess;
      };

      // inter-chunk seam: between the last block of the previous chunk and block bstart
      if (bstart > 0 && seam_tail_block == bstart - 1 && !seam_tail_buf.empty()) {
        size_t blk0_size = (size_t)ctx->h_output_sizes[s][0];
        size_t head_len  = std::min((size_t)(pattern_len - 1), blk0_size);
        if (!seam_tail_buf.empty() && seam_tail_buf.size() + head_len >= (size_t)pattern_len) {
          std::vector<uint8_t> seam(seam_tail_buf.size() + head_len);
          std::copy(seam_tail_buf.begin(), seam_tail_buf.end(), seam.begin());
          if (d2h(0, head_len, seam.data() + seam_tail_buf.size())) {
            cpu_search(seam.data(), seam.size(), cpat, pattern_len,
                       case_insensitive, seam_tail_base, result);
          }
        }
      }

      // intra-chunk seams: between adjacent blocks within this chunk
      for (int i = 0; i < bcount - 1; i++) {
        size_t blk_i_sz  = (size_t)ctx->h_output_sizes[s][i];
        size_t blk_i1_sz = (size_t)ctx->h_output_sizes[s][i + 1];
        size_t tail_len  = std::min((size_t)(pattern_len - 1), blk_i_sz);
        size_t head_len  = std::min((size_t)(pattern_len - 1), blk_i1_sz);
        if (tail_len + head_len < (size_t)pattern_len) continue;

        std::vector<uint8_t> seam(tail_len + head_len);
        size_t tail_dev = (size_t)i * max_block_size + blk_i_sz - tail_len;
        size_t head_dev = (size_t)(i + 1) * max_block_size;
        if (d2h(tail_dev, tail_len, seam.data()) &&
            d2h(head_dev, head_len, seam.data() + tail_len)) {
          size_t seam_base = prefix[bstart + i] + blk_i_sz - tail_len;
          cpu_search(seam.data(), seam.size(), cpat, pattern_len,
                     case_insensitive, seam_base, result);
        }
      }

      // save tail of this chunk's last block for the inter-chunk seam with the next chunk
      {
        int last = bcount - 1;
        size_t last_sz  = (size_t)ctx->h_output_sizes[s][last];
        size_t tail_len = std::min((size_t)(pattern_len - 1), last_sz);
        seam_tail_buf.resize(tail_len);
        if (tail_len > 0 && d2h((size_t)last * max_block_size + last_sz - tail_len,
                                 tail_len, seam_tail_buf.data())) {
          seam_tail_base  = prefix[bstart + last] + last_sz - tail_len;
          seam_tail_block = bstart + last;
        } else {
          seam_tail_buf.clear();
          seam_tail_block = -1;
        }
      }
    }

    pending[s].chunk_idx = -1;
    return true;
  };

  int num_chunks = (num_blocks + (int)blocks_per_chunk - 1) / (int)blocks_per_chunk;

  auto t_gpu_start = Clock::now();  // alloc + H2D done; GPU work starts here

  for (int ci = 0; ci < num_chunks && !error; ci++) {
    int s = ci % 2;

    // flush this slot — waits for chunk ci-2's kernel + D2H and processes results
    if (!flush_slot(s)) { error = true; break; }

    int bstart = ci * (int)blocks_per_chunk;
    int bcount = (int)std::min((size_t)(num_blocks - bstart), blocks_per_chunk);

    static constexpr int THREADS = 256;

    if (cudaMemsetAsync(ctx->d_match_count[s], 0, sizeof(int), ctx->streams[s]) != cudaSuccess) {
      error = true; break;
    }

    // one cuda block per lz4 block: thread 0 decompresses, then all
    // threads search. decompressed data is still hot in l2 when search starts
    decompress_and_search_kernel<<<bcount, THREADS, (size_t)pattern_len, ctx->streams[s]>>>(
        ctx->d_compressed, ctx->d_blocks + bstart, bcount,
        ctx->d_decompressed[s], ctx->d_output_sizes[s], max_block_size,
        ctx->d_pattern, pattern_len, case_insensitive,
        ctx->d_matches[s], ctx->d_match_count[s], (int)max_matches_per_chunk);
    if (cudaGetLastError() != cudaSuccess) { error = true; break; }

    // async D2H — all on the same stream so they're ordered after the kernel
    if (cudaMemcpyAsync(ctx->h_output_sizes[s], ctx->d_output_sizes[s],
                    (size_t)bcount * sizeof(int64_t), cudaMemcpyDeviceToHost, ctx->streams[s]) != cudaSuccess ||
        cudaMemcpyAsync(ctx->h_match_count[s], ctx->d_match_count[s],
                    sizeof(int), cudaMemcpyDeviceToHost, ctx->streams[s]) != cudaSuccess ||
        cudaMemcpyAsync(ctx->h_matches[s], ctx->d_matches[s],
                    max_matches_per_chunk * sizeof(size_t), cudaMemcpyDeviceToHost, ctx->streams[s]) != cudaSuccess) {
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
    cudaStreamSynchronize(ctx->streams[0]);
    cudaStreamSynchronize(ctx->streams[1]);
  }

  if (bench && !error) {
    auto t_end = Clock::now();
    double alloc_ms = Ms(t_gpu_start - t_frame_start).count();
    double gpu_ms   = Ms(t_end       - t_gpu_start).count();
    double decomp_mb = (double)prefix[num_blocks] / (1024.0 * 1024.0);
    double comp_mb   = (double)size / (1024.0 * 1024.0);
    fprintf(stderr, "[bench] alloc_ms=%.1f gpu_ms=%.1f decomp_mb=%.1f comp_mb=%.2f blocks=%d chunks=%d block_kb=%zu\n",
            alloc_ms, gpu_ms, decomp_mb, comp_mb, num_blocks, num_chunks, max_block_size / 1024);
  }

  if (error) return std::nullopt;
  if (decompressed_size_out) *decompressed_size_out = prefix[num_blocks];
  std::sort(result.begin(), result.end());
  return result;
}

std::optional<std::vector<size_t>> search_file(const uint8_t *data, size_t size,
                                                const std::string &pattern,
                                                GpuContext *ctx,
                                                bool case_insensitive,
                                                bool bench,
                                                bool verify_checksums) {
  std::vector<size_t> result;
  size_t pos = 0;
  size_t decompressed_base = 0;

  while (pos + 4 <= size) {
    // stop at anything that isn't an LZ4 frame magic number
    if (load_u32_le(data + pos) != FRAME_MAGIC_NUMBER) break;

    size_t consumed = 0, decomp_size = 0;
    auto frame_matches = search_frame(data + pos, size - pos, pattern,
                                      ctx, case_insensitive, &consumed, &decomp_size, bench,
                                      verify_checksums);
    if (!frame_matches) return std::nullopt;

    for (size_t off : *frame_matches)
      result.push_back(decompressed_base + off);

    decompressed_base += decomp_size;
    if (consumed == 0) break; // shouldn't happen, but guard against infinite loop
    pos += consumed;
  }

  return result;
}