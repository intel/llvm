// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3 -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, K define the sizes of dimensions of the three matrix types (a, b,
// accumulator) used per subgroup operation.
constexpr int M = 8; // number of rows of accumulator,
                     // number of cols of b.
constexpr int N = 8; // number of cols of accumulator,
                     // number of rows of a.
constexpr int K = 4; // number of cols of a/number of rows of b.

double A[M * K];
double B[K * N];
double C[M * N];
double D[M * N];

int main() {

  buffer<double, 1> bufA(A, range<1>(M * K));
  buffer<double, 1> bufB(B, range<1>(K * N));
  buffer<double, 1> bufC(C, range<1>(M * N));
  buffer<double, 1> bufD(D, range<1>(M * N));

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

          joint_matrix<double, matrix_use::accumulator, M, N,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<double, matrix_use::a, M, K, matrix_layout::row_major>
              sub_a;

          joint_matrix<double, matrix_use::b, K, N, matrix_layout::row_major>
              sub_b;

          //CHECK: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.load.c.row.stride.f64.p1f64(double addrspace(1)* %_arg_, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), N);
          //CHECK: tail call double @llvm.nvvm.wmma.m8n8k4.load.a.row.stride.f64.p1f64(double addrspace(1)* %_arg_4, i32 4) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), K);
          //CHECK: tail call double @llvm.nvvm.wmma.m8n8k4.load.b.row.stride.f64.p1f64(double addrspace(1)* %_arg_9, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), N);
          //CHECK: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.mma.row.row.f64(double %3, double %4, double %1, double %2) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          //CHECK: tail call void @llvm.nvvm.wmma.m8n8k4.store.d.row.stride.f64.p1f64(double addrspace(1)* %_arg_14, double %6, double %7, i32 8) #{{.*}}
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

          joint_matrix<double, matrix_use::accumulator, M, N,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<double, matrix_use::a, M, K, matrix_layout::col_major>
              sub_a;

          joint_matrix<double, matrix_use::b, K, N, matrix_layout::col_major>
              sub_b;

          //CHECK: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.load.c.col.stride.f64.p1f64(double addrspace(1)* %_arg_, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), M);
          //CHECK: tail call double @llvm.nvvm.wmma.m8n8k4.load.a.col.stride.f64.p1f64(double addrspace(1)* %_arg_4, i32 8) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), M);
          //CHECK: tail call double @llvm.nvvm.wmma.m8n8k4.load.b.col.stride.f64.p1f64(double addrspace(1)* %_arg_9, i32 4) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), K);
          //CHECK: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.mma.col.col.f64(double %3, double %4, double %1, double %2) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          //CHECK: tail call void @llvm.nvvm.wmma.m8n8k4.store.d.col.stride.f64.p1f64(double addrspace(1)* %_arg_14, double %6, double %7, i32 8) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), M);
        });
  });

  return 0;
};
