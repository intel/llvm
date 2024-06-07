// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

// IMPORTANT: before updating sm version support beyond sm_90 read the following
// NOTE!

// NOTE: Technically the 'wrong' ptx instruction is called by
// joint_matrix_load/joint_matrix_store in this case: notice that the load and
// store instructions use shape m16n16k16, rather than the correct shape
// m16n16k8. The 'wrong' ptx instruction is used because it returns the correct
// SASS instructions for all existing sm versions supporting tf32: sm_80, sm_86,
// sm_87, sm_89, and sm_90. The reason for this ptx instruction redundancy is
// due to the ptx naming convention for the mnk shape triple; however we cannot
// in principle a priori know that future sm versions will behave in the same
// way and that this redundancy will continue as future architecture is
// released. This should be validated before supporting any sm versions beyond
// sm_90. The reason that we choose to use the m16n16k16 instruction is that it
// allows us to use a simpler portable interface across Intel and Nvidia
// backends.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, K define the sizes of dimensions of the three matrix types (a, b,
// accumulator) used per subgroup operation.
constexpr int M = 16; // number of rows of accumulator,
                      // number of cols of b.
constexpr int N = 16; // number of cols of accumulator,
                      // number of rows of a.
constexpr int K = 8;  // number of cols of a/number of rows of b.

// Float is used in this test as the storage type for tf32:
//
// float A[M * K];
// float B[K * N];
// float C[M * N];
// float D[M * N];
//
// Accessors would have been made, like so:
//
// buffer<float, 1> bufA(A, range<1>(M * K)); // will be used as tf32
// buffer<float, 1> bufB(B, range<1>(K * N)); // will be used as tf32
// buffer<float, 1> bufC(C, range<1>(M * N));
// buffer<float, 1> bufD(D, range<1>(M * N));
// ...
// auto accA = bufA.get_access<access::mode::read_write>(handler);
// auto accB = bufB.get_access<access::mode::read_write>(handler);
// auto accC = bufC.get_access<access::mode::read_write>(handler);
// auto accD = bufD.get_access<access::mode::read_write>(handler);

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
row_row(sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accA,
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accB,
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accC,
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accD,
        nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, precision::tf32, use::a, M, K, layout::row_major>
      sub_a{};
  joint_matrix<sub_group, precision::tf32, use::b, K, N, layout::row_major>
      sub_b{};
  joint_matrix<sub_group, float, use::accumulator, M, N> sub_c{};

  //CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0(ptr %{{.*}}, i32 8)
  joint_matrix_load(sg, sub_a,
                    accA.template get_multi_ptr<access::decorated::yes>(), K);
  //CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.row.stride.tf32.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_b,
                    accB.template get_multi_ptr<access::decorated::yes>(), N);
  //CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(), N,
                    layout::row_major);

  auto round_lambda = [](auto &x) { x = round_to_tf32(x); };
  //CHECK-OPAQUE: tail call i32 @llvm.nvvm.f2tf32.rna(float %{{.*}})
  joint_matrix_apply(sg, sub_a, round_lambda);

  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  //CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1(ptr addrspace(1) {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 {{.*}}
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), N,
                     layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
col_col(sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accA,
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accB,
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accC,
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::target::device>
            accD,
        nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, precision::tf32, use::a, M, K, layout::col_major>
      sub_a{};
  joint_matrix<sub_group, precision::tf32, use::b, K, N, layout::col_major>
      sub_b{};
  joint_matrix<sub_group, float, use::accumulator, M, N> sub_c{};

  //CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.col.stride.tf32.p0(ptr %{{.*}}, i32 8)
  joint_matrix_load(sg, sub_a,
                    accA.template get_multi_ptr<access::decorated::yes>(), K);
  //CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.col.stride.tf32.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_b,
                    accB.template get_multi_ptr<access::decorated::yes>(), N);
  //CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p1(ptr addrspace(1) {{.*}}, i32 {{.*}})
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(), N,
                    layout::col_major);

  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  //CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1(ptr addrspace(1) {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), N,
                     layout::col_major);
}
