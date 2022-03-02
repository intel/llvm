// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3 -S -Xclang -emit-llvm %s -o -| FileCheck %s

// IMPORTANT: before updating sm version support beyond sm_86 read the following
// NOTE!

// NOTE: Technically the 'wrong' ptx instruction is called by
// joint_matrix_load/joint_matrix_store in this case: notice that the load and
// store instructions use shape m16n16k16, rather than the correct shape
// m16n16k8. The 'wrong' ptx instruction is used because it returns the correct
// SASS instructions for all existing supported sm versions: sm_80 and sm_86.
// The reason for this ptx instruction redundancy is due to the ptx naming
// convention for the mnk shape triple; however we cannot in principle a priori
// know that future sm versions will behave in the same way and that this
// redundancy will continue as future architecture is released. This should be
// validated before supporting any sm versions beyond sm_86. The reason that we
// choose to use the m16n16k16 instruction is that it allows the significant
// advantage of being able to use a portable interface across Intel and Nvidia
// backends.

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, K define the sizes of dimensions of the three matrix types (a, b,
// accumulator) used per subgroup operation.
constexpr int M = 16; // number of rows of accumulator,
                      // number of cols of b.
constexpr int N = 16; // number of cols of accumulator,
                      // number of rows of a.
constexpr int K = 8;  // number of cols of a/number of rows of b.

uint32_t A[M * K];
uint32_t B[K * N];
float C[M * N];
float D[M * N];

int main() {

  buffer<uint32_t, 1> bufA(A, range<1>(M * K));
  buffer<uint32_t, 1> bufB(B, range<1>(K * N));
  buffer<float, 1> bufC(C, range<1>(M * N));
  buffer<float, 1> bufD(D, range<1>(M * N));

  queue q;

  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);
    auto accD = bufD.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class row_row>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<float, matrix_use::accumulator, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<uint32_t, matrix_use::a, M, K, matrix_layout::row_major>
              sub_a;

          joint_matrix<uint32_t, matrix_use::b, K, N, matrix_layout::row_major>
              sub_b;
 
          //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p1f32(float addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), N);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), K);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.row.stride.tf32.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), N);
          //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32 %10, i32 %11, i32 %12, i32 %13, i32 %15, i32 %16, i32 %17, i32 %18, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          //CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1f32(float addrspace(1)* %_arg_14, float %20, float %21, float %22, float %23, float %24, float %25, float %26, float %27, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), N);
        });
  });

  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);
    auto accD = bufD.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class col_col>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<float, matrix_use::accumulator, M, N,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<uint32_t, matrix_use::a, M, K, matrix_layout::col_major>
              sub_a;

          joint_matrix<uint32_t, matrix_use::b, K, N, matrix_layout::col_major>
              sub_b;

          //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p1f32(float addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), N);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.col.stride.tf32.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), K);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.col.stride.tf32.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), N);
          //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.col.col.tf32(i32 %10, i32 %11, i32 %12, i32 %13, i32 %15, i32 %16, i32 %17, i32 %18, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          //CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1f32(float addrspace(1)* %_arg_14, float %20, float %21, float %22, float %23, float %24, float %25, float %26, float %27, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), N);
        });
  });

  return 0;
};
