// RUN: %clangxx -fsycl -fsycl-device-only -O2 -S -emit-llvm -o - %s | FileCheck %s

// CHECK-DAG: target("spirv.JointMatrixINTEL", i8, 12, 48, 0, 3, 0)
// CHECK-DAG: target("spirv.JointMatrixINTEL", i32, 12, 12, 3, 3, 2)
// CHECK-DAG: target("spirv.JointMatrixINTEL", i8, 48, 12, 2, 3, 1)

// CHECK: !{!"matrix_type::sint32,use::accumulator,12,12;matrix_type::sint8,use::a,12,48;matrix_type::sint8,use::b,48,12"}
// CHECK: !{!"matrix_type::sint8,matrix_type::sint8,matrix_type::sint32,matrix_type::sint32,12,48,12"}

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define TILE_SZ 16
#define TM (TILE_SZ - 4)
#define TN (TILE_SZ - 4)
#define TK (4 * TILE_SZ - 16)

#define SG_SZ 16

// static constexpr size_t MATRIX_M = TM * 2;
// static constexpr size_t MATRIX_N = TN * 2;
// static constexpr size_t MATRIX_K = TK * 2;
// int8_t A[MATRIX_M][MATRIX_K];
// int8_t B[MATRIX_K / 4][MATRIX_N * 4];
// int32_t C[MATRIX_M][MATRIX_N];

SYCL_EXTERNAL [[intel::reqd_sub_group_size(SG_SZ)]] void
matrix_multiply(size_t NUM_COLS_C, size_t NUM_COLS_A,
                sycl::accessor<int8_t, 2, access::mode::read_write> accA,
                sycl::accessor<int8_t, 2, access::mode::read_write> accB,
                sycl::accessor<int32_t, 2, access::mode::read_write> accC,
                nd_item<2> spmd_item) {

  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  // The submatrix API has to be accessed by all the workitems in a
  // subgroup these functions will be called once by the subgroup no
  // code divergence between the workitems
  const auto global_idx = spmd_item.get_global_id(0);
  const auto global_idy = spmd_item.get_global_id(1);
  const auto sg_startx = global_idx - spmd_item.get_local_id(0);
  const auto sg_starty = global_idy - spmd_item.get_local_id(1);

  sycl::sub_group sg = spmd_item.get_sub_group();
  joint_matrix<sycl::sub_group, int8_t, use::a, TM, TK, layout::row_major>
      sub_a;
  // For B, since current implementation does not support non-packed
  // layout, users need to specify the updated VNNI sizes along with
  // the packed_b layout. By default, the layout is row_major and size
  // is (TK, TN).
  joint_matrix<sycl::sub_group, int8_t, use::b, TK, TN,
               layout::ext_intel_packed>
      sub_b;
  joint_matrix<sycl::sub_group, int32_t, use::accumulator, TM, TN> sub_c;

  // AMX: 8 register tiles : 1k byte size, SMmaxxSKmax =16x64
  // strideX = X's cols, so strideC = N, strideA = K, strideB = N*4
  joint_matrix_fill(sg, sub_c, 0);
  for (int k = 0; k < K / TK; k += 1) {
    joint_matrix_load(
        sg, sub_a,
        accA.template get_multi_ptr<sycl::access::decorated::no>() +
            (sg_startx * TM) * K + k * TK,
        K);
    // Assuming B data is already in VNNI format.
    joint_matrix_load(
        sg, sub_b,
        accB.template get_multi_ptr<sycl::access::decorated::no>() +
            (k * TK / 4) * (N * 4) + sg_starty / SG_SZ * TN * 4,
        N * 4);
    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  }
  joint_matrix_store(
      sg, sub_c,
      accC.template get_multi_ptr<sycl::access::decorated::no>() +
          (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
      N, layout::row_major);
}
