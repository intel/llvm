// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// M, N, K define the sizes of dimensions of the three matrix types (a, b,
// accumulator) used per subgroup operation.
constexpr int M = 8; // number of rows of accumulator,
                     // number of cols of b.
constexpr int N = 8; // number of cols of accumulator,
                     // number of rows of a.
constexpr int K = 4; // number of cols of a/number of rows of b.

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
row_row_m8n8k4(sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accA,
               sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accB,
               sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accC,
               sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accD,
               nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, double, use::accumulator, M, N> sub_c{};
  joint_matrix<sub_group, double, use::a, M, K, layout::row_major> sub_a{};
  joint_matrix<sub_group, double, use::b, K, N, layout::row_major> sub_b{};

  //CHECK-OPAQUE: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.load.c.row.stride.f64.p1(ptr addrspace(1) %{{.*}}, i32 8)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(), N,
                    layout::row_major);
  //CHECK-OPAQUE: tail call double @llvm.nvvm.wmma.m8n8k4.load.a.row.stride.f64.p1(ptr addrspace(1) %{{.*}}, i32 4)
  joint_matrix_load(sg, sub_a,
                    accA.template get_multi_ptr<access::decorated::yes>(), K);
  //CHECK-OPAQUE: tail call double @llvm.nvvm.wmma.m8n8k4.load.b.row.stride.f64.p1(ptr addrspace(1) %{{.*}}, i32 8)
  joint_matrix_load(sg, sub_b,
                    accB.template get_multi_ptr<access::decorated::yes>(), N);
  //CHECK-OPAQUE: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.mma.row.row.f64(double {{.*}}, double {{.*}}, double {{.*}}, double {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  //CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n8k4.store.d.row.stride.f64.p1(ptr addrspace(1) %{{.*}}, double {{.*}}, double {{.*}}, i32 8)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), N,
                     layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
col_col_m8n8k4(sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accA,
               sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accB,
               sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accC,
               sycl::accessor<double, 1, sycl::access::mode::read_write,
                              sycl::target::device>
                   accD,
               nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, double, use::accumulator, M, N> sub_c{};
  joint_matrix<sub_group, double, use::a, M, K, layout::col_major> sub_a{};
  joint_matrix<sub_group, double, use::b, K, N, layout::col_major> sub_b{};

  //CHECK-OPAQUE: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.load.c.col.stride.f64.p1(ptr addrspace(1) %{{.*}}, i32 8)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(), M,
                    layout::col_major);
  //CHECK-OPAQUE: tail call double @llvm.nvvm.wmma.m8n8k4.load.a.col.stride.f64.p1(ptr addrspace(1) %{{.*}}, i32 8)
  joint_matrix_load(sg, sub_a,
                    accA.template get_multi_ptr<access::decorated::yes>(), M);
  //CHECK-OPAQUE: tail call double @llvm.nvvm.wmma.m8n8k4.load.b.col.stride.f64.p1(ptr addrspace(1) %{{.*}}, i32 4)
  joint_matrix_load(sg, sub_b,
                    accB.template get_multi_ptr<access::decorated::yes>(), K);
  //CHECK-OPAQUE: tail call { double, double } @llvm.nvvm.wmma.m8n8k4.mma.col.col.f64(double {{.*}}, double {{.*}}, double {{.*}}, double {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  //CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n8k4.store.d.col.stride.f64.p1(ptr addrspace(1) %{{.*}}, double {{.*}}, double {{.*}}, i32 8)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), M,
                     layout::col_major);
}
