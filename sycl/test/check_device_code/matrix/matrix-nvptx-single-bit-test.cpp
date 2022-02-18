// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3 -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, (K * 32) define the sizes of dimensions of the three matrix types (a,
// b, accumulator) used per subgroup operation.
constexpr int M = 8; // number of rows of accumulator,
                     // number of cols of b.
constexpr int N = 8; // number of cols of accumulator,
                     // number of rows of a.
constexpr int K = 4; // number of cols of a/number of rows of b divided by 32

// Each bit of each uint32_t A/B array element is an element of a single-bit
// matrix. joint_matrix_bmad performs Binary Dot Products on these matrices (see
// M. Rastegari et al. Computer Vision – ECCV 2016, 525-542 and A. Li et al.
// IEEE Transactions on Parallel and Distributed Systems, 32(7):1878-1891,
// 2021))
uint32_t A[M * K];
uint32_t B[K * N];
int32_t C[M * N];
int32_t D[M * N];

int main() {

  buffer<uint32_t, 1> bufA(A, range<1>(M * K));
  buffer<uint32_t, 1> bufB(B, range<1>(K * N));
  buffer<int32_t, 1> bufC(C, range<1>(M * N));
  buffer<int32_t, 1> bufD(D, range<1>(M * N));

  queue q;

  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);
    auto accD = bufD.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class row_col>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<uint32_t, matrix_use::a, M, K, matrix_layout::row_major>
              sub_a;

          joint_matrix<uint32_t, matrix_use::b, K, N, matrix_layout::col_major>
              sub_b;

          //CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m8n8k128.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), N);
          //CHECK: tail call i32 @llvm.nvvm.wmma.m8n8k128.load.a.row.stride.b1.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 128) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), K);
          //CHECK: tail call i32 @llvm.nvvm.wmma.m8n8k128.load.b.col.stride.b1.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 128) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), K);
          //CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m8n8k128.mma.xor.popc.row.col.b1(i32 %3, i32 %4, i32 %1, i32 %2) #{{.*}}
          sub_c = joint_matrix_bmad(sg, sub_a, sub_b, sub_c,
                                    sycl::bit_xor<uint32_t>());
          //CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m8n8k128.mma.and.popc.row.col.b1(i32 %3, i32 %4, i32 %6, i32 %7) #{{.*}}
          sub_c = joint_matrix_bmad(sg, sub_a, sub_b, sub_c,
                                    sycl::bit_and<uint32_t>());
          //CHECK: tail call void @llvm.nvvm.wmma.m8n8k128.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %9, i32 %10, i32 8) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), N);
        });
  });

  return 0;
};
