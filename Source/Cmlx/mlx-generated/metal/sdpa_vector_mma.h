// Copyright © 2026 Apple Inc.

#pragma once

#include "steel/attn/attn.h"

struct SdpaMmaMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SdpaMmaSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct SdpaMmaMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SdpaMmaExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

// First pass of the 2-pass vector attention for 2 to 8 queries, as a
// speculative-decoding verify pass sends. sdpa_vector_2pass_1 gives each
// (query head, query position) row its own simdgroup, so every row reads the
// whole key and value range and the cost grows with the row count. Here a
// threadgroup takes all the rows of one KV head (gqa_factor * q_seq_len of
// them, contiguous in a row-contiguous query) and one contiguous chunk of the
// keys, stages each key/value tile in threadgroup memory once, and runs
// Q @ K.T and P @ V on simdgroup matrices. The partials layout is
// sdpa_vector_2pass_1's, so sdpa_vector_2pass_2 combines them unchanged.
//
// No array mask and no sinks; do_causal masks the last q_seq_len keys by
// query position. Each simdgroup owns 8 rows; WM * 8 must cover the rows, and
// WM is 1, 2 or 4 so the tile loaders' thread count divides the tiles.
template <typename T, int D, int V, int WM, int BK = 16>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void
sdpa_vector_2pass_1_mma(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    const constant size_t& v_head_stride [[buffer(10)]],
    const constant size_t& v_seq_stride [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    const constant int& n_rows [[buffer(19)]],
    const constant int& q_seq_len [[buffer(20)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]]) {
  using namespace mlx::steel;
  typedef float AccumType;
  static_assert(WM == 1 || WM == 2 || WM == 4, "WM must be 1, 2 or 4");

  constexpr short BQ = WM * 8;
  constexpr short kFragSize = 8;
  constexpr short TK = BK / kFragSize;
  constexpr short TD = D / kFragSize;
  constexpr short TV = V / kFragSize;

  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int num_kv_heads = tpg.x;
  const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
  // Row r of this KV head is query head kv * gqa + r / q_seq_len at position
  // r % q_seq_len, and its output offset is row_base + r.
  const int row_base = kv_batch_head_idx * n_rows;

  // This block's contiguous chunk of keys, a whole number of tiles.
  const int chunk = (((N + blocks - 1) / blocks) + BK - 1) / BK * BK;
  const int kstart = min(N, block_idx * chunk);
  const int kend = min(N, kstart + chunk);

  queries += row_base * D;
  keys += kv_batch_head_idx * k_head_stride + kstart * k_seq_stride;
  values += kv_batch_head_idx * v_head_stride + kstart * v_seq_stride;

  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDQ_tgp = D + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = V + padV;
  constexpr short tgp_mem_k = (BK + padK) * D;
  constexpr short tgp_mem_v = BK * (V + padV);
  constexpr short tgp_mem_s = tgp_mem_k > tgp_mem_v ? tgp_mem_k : tgp_mem_v;

  threadgroup T Qs[BQ * LDQ_tgp];
  threadgroup T KV_smem[tgp_mem_s];
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using QBlockLoader = BlockLoaderT<T, BQ, D, LDQ_tgp, 1, 1, WM * 32>;
  // K is loaded transposed.
  using KBlockLoader = BlockLoaderT<T, BK, D, 1, LDK_tgp, 0, WM * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, V, LDV_tgp, 1, 0, WM * 32>;

  QBlockLoader loader_q(queries, D, Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(keys, k_seq_stride, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(values, v_seq_stride, Vs, simd_group_id, simd_lane_id);

  const AccumType scale_log2 = scale * M_LOG2E_F;

  using MMAFrag_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, 1, 1, MMAFrag_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_t> Ktile;
  MMATile<AccumType, 1, TK, MMAFrag_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_t> Vtile;
  MMATile<AccumType, 1, TV, MMAFrag_t> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFrag_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  loader_q.load_safe(short2(D, n_rows));

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  static_assert(kRowsPT == 1, "one row per thread");
  AccumType max_score = Limits<AccumType>::finite_min;
  AccumType sum_score = 0;

  // The query position of this thread's row, for the causal limit.
  const int row = tm + sm;
  const int causal_limit = N - q_seq_len + (row % q_seq_len);

  for (int k0 = kstart; k0 < kend; k0 += BK) {
    const short k_rem = min(int(BK), kend - k0);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_rem < BK) {
      loader_k.load_safe(short2(D, k_rem));
    } else {
      loader_k.load_unsafe();
    }

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * kFragSize]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * kFragSize * LDK_tgp]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale_log2;
    }

    // Keys past the chunk, and keys after this row's position.
    const bool tail = k_rem < BK;
    const bool causal_tile = do_causal && (k0 + BK - 1) > (N - q_seq_len);
    if (tail || causal_tile) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TK; j++) {
        const short col = sn + j * kFragSize;
        STEEL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
          const int key = k0 + col + jj;
          if ((col + jj) >= k_rem || (do_causal && key > causal_limit)) {
            Stile.frag_at(0, j)[jj] = neg_inf;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_rem < BK) {
      loader_v.load_safe(short2(V, k_rem));
    } else {
      loader_v.load_unsafe();
    }

    AccumType new_max = max_score;
    Stile.template row_reduce<SdpaMmaMaxOp>(&new_max);
    Stile.template row_bin_op<SdpaMmaExpSubOp>(&new_max);
    AccumType factor = fast::exp2(max_score - new_max);
    max_score = new_max;
    AccumType tile_sum = 0;
    Stile.template row_reduce<SdpaMmaSumOp>(&tile_sum);
    sum_score = sum_score * factor + tile_sum;
    Otile.template row_bin_op<SdpaMmaMulOp>(&factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TV; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        simdgroup_barrier(mem_flags::mem_none);
        Vtile.template load<T, 1, 1, LDV_tgp, 1>(
            &Vs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
        simdgroup_barrier(mem_flags::mem_none);
        MMAFrag_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }

    loader_k.next();
    loader_v.next();
  }

  // Partials: the unnormalized output, its sum and its max (natural log).
  if (row < n_rows) {
    device T* o = out + (row_base + row) * blocks * V + block_idx * V + sn;
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TV; id++) {
      STEEL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_t::kElemCols; jj++) {
        o[id * kFragSize + jj] = static_cast<T>(Otile.frag_at(0, id)[jj]);
      }
    }
    if (sn == 0) {
      sums[(row_base + row) * blocks + block_idx] = sum_score;
      maxs[(row_base + row) * blocks + block_idx] = max_score * M_LN2_F;
    }
  }
}
