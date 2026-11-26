// REQUIRES: hip
// REQUIRES: hip-arch-gfx90a || hip-arch-gfx940 || hip-arch-gfx941 || hip-arch-gfx942 || hip-arch-gfx950
// RUN: %if hip-arch-gfx90a %{ %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s --check-prefix=CHECK-CDNA2 %}
// RUN: %if hip-arch-gfx940 || hip-arch-gfx941 || hip-arch-gfx942 %{ %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s --check-prefix=CHECK-CDNA3 %}
// RUN: %if hip-arch-gfx950 %{ %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s --check-prefixes=CHECK-CDNA3,CHECK-CDNA4 %}

// gfx90a (CDNA2) int8 MFMAs are 16x16x16 and 32x32x8 with i32 operands.
// CDNA3 (gfx940, gfx941, gfx942) replaces them with 16x16x32 and 32x32x16,
// with the A/B operands packed into i64 values, and gfx950 (CDNA4) adds
// 16x16x64 and 32x32x32 with 128-bit operands.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#if defined(__gfx90a__)
SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_row_m16n16k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::row_major> sub_b{};

  // CHECK-CDNA2: tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 {{.*}}, i32 {{.*}}, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 16,
                     layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_col_m32n32k8(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accA,
                 sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accB,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accC,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accD,
                 nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 32, 32> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 32, 8, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 8, 32, layout::col_major> sub_b{};

  // CHECK-CDNA2: tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 {{.*}}, i32 {{.*}}, <16 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 32,
                     layout::row_major);
}
#else
SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_row_m16n16k32(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 16, 32, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 32, 16, layout::row_major> sub_b{};

  // CHECK-CDNA3: tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 {{.*}}, i64 {{.*}}, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 16,
                     layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_col_m32n32k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 32, 32> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 32, 16, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 32, layout::col_major> sub_b{};

  // CHECK-CDNA3: tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x16.i8(i64 {{.*}}, i64 {{.*}}, <16 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 32,
                     layout::row_major);
}
#endif

#if defined(__gfx950__)
SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_row_m16n16k64(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 16, 64, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 64, 16, layout::row_major> sub_b{};

  // CHECK-CDNA4: tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x64.i8(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 16,
                     layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
row_col_m32n32k32(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 32, 32> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 32, 32, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 32, 32, layout::col_major> sub_b{};

  // CHECK-CDNA4: tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x32.i8(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <16 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), 32,
                     layout::row_major);
}
#endif
