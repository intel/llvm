// REQUIRES: hip
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amd_gpu_gfx90a -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_row_m16n16k16(sycl::accessor<half, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<half, 1, sycl::access::mode::read_write,
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

  joint_matrix<sub_group, float, use::accumulator, 16, 16> sub_c{};
  joint_matrix<sub_group, half, use::a, 16, 16, layout::row_major> sub_a{};
  joint_matrix<sub_group, half, use::b, 16, 16, layout::row_major> sub_b{};

  // CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 16,
                     layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_col_m32n32k8(sycl::accessor<half, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accA,
                 sycl::accessor<half, 1, sycl::access::mode::read_write,
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

  joint_matrix<sub_group, float, use::accumulator, 32, 32> sub_c{};
  joint_matrix<sub_group, half, use::a, 32, 8, layout::row_major> sub_a{};
  joint_matrix<sub_group, half, use::b, 8, 32, layout::col_major> sub_b{};

  // CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 32,
                     layout::row_major);
}