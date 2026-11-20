// REQUIRES: hip
// REQUIRES: hip-arch-gfx950
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s

// OCP FP8 multiplicands use the FP8 MFMAs (16x16x32, 32x32x16), whose opcode
// selects the A/B formats, and the F8F6F4 MFMAs (16x16x128, 32x32x64), whose
// CBSZ/BLGP operands select them (0 = E4M3, 1 = E5M2).

#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::experimental::fp8_e4m3;
using sycl::ext::oneapi::experimental::fp8_e5m2;

template <typename TA, typename TB, size_t M, size_t N, size_t K>
void mad(TA *A, TB *B, float *D, sub_group sg) {
  joint_matrix<sub_group, float, use::accumulator, M, N> sub_c{};
  joint_matrix<sub_group, TA, use::a, M, K, layout::row_major> sub_a{};
  joint_matrix<sub_group, TB, use::b, K, N, layout::col_major> sub_b{};
  joint_matrix_load(sg, sub_a,
                    address_space_cast<access::address_space::global_space,
                                       access::decorated::no>(A),
                    K);
  joint_matrix_load(sg, sub_b,
                    address_space_cast<access::address_space::global_space,
                                       access::decorated::no>(B),
                    K);
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  joint_matrix_store(sg, sub_c,
                     address_space_cast<access::address_space::global_space,
                                        access::decorated::no>(D),
                     N, layout::row_major);
}

#define TEST(NAME, TA, TB, M, N, K)                                            \
  SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void NAME(            \
      TA *A, TB *B, float *D, nd_item<2> item) {                               \
    mad<TA, TB, M, N, K>(A, B, D, item.get_sub_group());                       \
  }

// CHECK-LABEL: e4m3_e4m3_m16n16k32
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e4m3_e4m3_m16n16k32, fp8_e4m3, fp8_e4m3, 16, 16, 32)
// CHECK-LABEL: e4m3_e5m2_m16n16k32
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e4m3_e5m2_m16n16k32, fp8_e4m3, fp8_e5m2, 16, 16, 32)
// CHECK-LABEL: e5m2_e4m3_m32n32k16
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e5m2_e4m3_m32n32k16, fp8_e5m2, fp8_e4m3, 32, 32, 16)
// CHECK-LABEL: e5m2_e5m2_m32n32k16
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e5m2_e5m2_m32n32k16, fp8_e5m2, fp8_e5m2, 32, 32, 16)

// CHECK-LABEL: e4m3_e5m2_m16n16k128
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> {{.*}}, <8 x i32> {{.*}}, <4 x float> zeroinitializer, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
TEST(e4m3_e5m2_m16n16k128, fp8_e4m3, fp8_e5m2, 16, 16, 128)
// CHECK-LABEL: e5m2_e4m3_m32n32k64
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> {{.*}}, <8 x i32> {{.*}}, <16 x float> zeroinitializer, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
TEST(e5m2_e4m3_m32n32k64, fp8_e5m2, fp8_e4m3, 32, 32, 64)
