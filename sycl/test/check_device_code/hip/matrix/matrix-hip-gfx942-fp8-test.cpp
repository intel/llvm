// REQUIRES: hip
// REQUIRES: hip-arch-gfx942 || hip-arch-gfx941 || hip-arch-gfx940
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s

// gfx942 (CDNA3) fp8 (E4M3) / bf8 (E5M2) MFMA: 16x16x32 and 32x32x16 with the
// A/B operands packed into i64 values and an f32 accumulator. The A and B
// operand formats are independent, so all four format pairs are exercised for
// both shapes.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::experimental::fp8_e4m3;
using sycl::ext::oneapi::experimental::fp8_e5m2;

template <typename TA, typename TB, size_t M, size_t N, size_t K>
static void
mad(sycl::sub_group sg,
    sycl::accessor<TA, 1, sycl::access::mode::read_write, sycl::target::device>
        accA,
    sycl::accessor<TB, 1, sycl::access::mode::read_write, sycl::target::device>
        accB,
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD) {
  joint_matrix<sub_group, float, use::accumulator, M, N> sub_c{};
  joint_matrix<sub_group, TA, use::a, M, K, layout::row_major> sub_a{};
  joint_matrix<sub_group, TB, use::b, K, N, layout::row_major> sub_b{};
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(), N,
                     layout::row_major);
}

// --- 16x16x32 -------------------------------------------------------------

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e4m3_e4m3_m16n16k32(sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e4m3, fp8_e4m3, 16, 16, 32>(item.get_sub_group(), accA, accB, accD);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e4m3_e5m2_m16n16k32(sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e4m3, fp8_e5m2, 16, 16, 32>(item.get_sub_group(), accA, accB, accD);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e5m2_e4m3_m16n16k32(sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e5m2, fp8_e4m3, 16, 16, 32>(item.get_sub_group(), accA, accB, accD);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e5m2_e5m2_m16n16k32(sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e5m2, fp8_e5m2, 16, 16, 32>(item.get_sub_group(), accA, accB, accD);
}

// --- 32x32x16 -------------------------------------------------------------

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e4m3_e4m3_m32n32k16(sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e4m3, fp8_e4m3, 32, 32, 16>(item.get_sub_group(), accA, accB, accD);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e4m3_e5m2_m32n32k16(sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e4m3, fp8_e5m2, 32, 32, 16>(item.get_sub_group(), accA, accB, accD);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e5m2_e4m3_m32n32k16(sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e4m3, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e5m2, fp8_e4m3, 32, 32, 16>(item.get_sub_group(), accA, accB, accD);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void
e5m2_e5m2_m32n32k16(sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accA,
                    sycl::accessor<fp8_e5m2, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accB,
                    sycl::accessor<float, 1, sycl::access::mode::read_write,
                                   sycl::target::device>
                        accD,
                    nd_item<2> item) {
  // CHECK: call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
  mad<fp8_e5m2, fp8_e5m2, 32, 32, 16>(item.get_sub_group(), accA, accB, accD);
}
