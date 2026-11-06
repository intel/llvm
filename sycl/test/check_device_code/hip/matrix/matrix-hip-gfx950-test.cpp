// REQUIRES: hip
// REQUIRES: hip-arch-gfx950
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s

// gfx950 (CDNA4) MFMAs added on top of the gfx942 ones: double-K half/bfloat16
// (16x16x32, 32x32x16) and int8 (16x16x64, 32x32x32) shapes, and the OCP FP8
// and FP4 multiplicands, which use the FP8 MFMAs (16x16x32, 32x32x16) and the
// F8F6F4 MFMAs (16x16x128, 32x32x64) whose CBSZ/BLGP operands select the A/B
// formats (0 = E4M3, 1 = E5M2, 4 = E2M1).

#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;
using sycl::ext::oneapi::experimental::fp4_e2m1;
using sycl::ext::oneapi::experimental::fp8_e4m3;
using sycl::ext::oneapi::experimental::fp8_e5m2;

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K>
void mad(TA *A, TB *B, TC *D, sub_group sg) {
  joint_matrix<sub_group, TC, use::accumulator, M, N> sub_c{};
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

#define TEST(NAME, TA, TB, TC, M, N, K)                                        \
  SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 64)]] void NAME(            \
      TA *A, TB *B, TC *D, nd_item<2> item) {                                  \
    mad<TA, TB, TC, M, N, K>(A, B, D, item.get_sub_group());                   \
  }

// CHECK-LABEL: half_m16n16k32
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> {{.*}}, <8 x half> {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(half_m16n16k32, half, half, float, 16, 16, 32)
// CHECK-LABEL: half_m32n32k16
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half> {{.*}}, <8 x half> {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(half_m32n32k16, half, half, float, 32, 32, 16)
// CHECK-LABEL: bf16_m16n16k32
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat> {{.*}}, <8 x bfloat> {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(bf16_m16n16k32, bfloat16, bfloat16, float, 16, 16, 32)
// CHECK-LABEL: bf16_m32n32k16
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat> {{.*}}, <8 x bfloat> {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(bf16_m32n32k16, bfloat16, bfloat16, float, 32, 32, 16)
// CHECK-LABEL: int8_m16n16k64
// CHECK: tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x64.i8(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
TEST(int8_m16n16k64, int8_t, int8_t, int32_t, 16, 16, 64)
// CHECK-LABEL: int8_m32n32k32
// CHECK: tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x32.i8(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <16 x i32> zeroinitializer, i32 0, i32 0, i32 0)
TEST(int8_m32n32k32, int8_t, int8_t, int32_t, 32, 32, 32)

// The gfx942 shapes are still available.
// CHECK-LABEL: half_m16n16k16
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> {{.*}}, <4 x half> {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(half_m16n16k16, half, half, float, 16, 16, 16)
// CHECK-LABEL: int8_m16n16k32
// CHECK: tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 {{.*}}, i64 {{.*}}, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
TEST(int8_m16n16k32, int8_t, int8_t, int32_t, 16, 16, 32)

// CHECK-LABEL: e4m3_e4m3_m16n16k32
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e4m3_e4m3_m16n16k32, fp8_e4m3, fp8_e4m3, float, 16, 16, 32)
// CHECK-LABEL: e4m3_e5m2_m16n16k32
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64 {{.*}}, i64 {{.*}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e4m3_e5m2_m16n16k32, fp8_e4m3, fp8_e5m2, float, 16, 16, 32)
// CHECK-LABEL: e5m2_e4m3_m32n32k16
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e5m2_e4m3_m32n32k16, fp8_e5m2, fp8_e4m3, float, 32, 32, 16)
// CHECK-LABEL: e5m2_e5m2_m32n32k16
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 {{.*}}, i64 {{.*}}, <16 x float> zeroinitializer, i32 0, i32 0, i32 0)
TEST(e5m2_e5m2_m32n32k16, fp8_e5m2, fp8_e5m2, float, 32, 32, 16)

// CHECK-LABEL: e4m3_e5m2_m16n16k128
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> {{.*}}, <8 x i32> {{.*}}, <4 x float> zeroinitializer, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0)
TEST(e4m3_e5m2_m16n16k128, fp8_e4m3, fp8_e5m2, float, 16, 16, 128)
// CHECK-LABEL: e5m2_e4m3_m32n32k64
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32(<8 x i32> {{.*}}, <8 x i32> {{.*}}, <16 x float> zeroinitializer, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
TEST(e5m2_e4m3_m32n32k64, fp8_e5m2, fp8_e4m3, float, 32, 32, 64)
// CHECK-LABEL: e2m1_e2m1_m16n16k128
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
TEST(e2m1_e2m1_m16n16k128, fp4_e2m1, fp4_e2m1, float, 16, 16, 128)
// CHECK-LABEL: e2m1_e4m3_m32n32k64
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> {{.*}}, <8 x i32> {{.*}}, <16 x float> zeroinitializer, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
TEST(e2m1_e4m3_m32n32k64, fp4_e2m1, fp8_e4m3, float, 32, 32, 64)
// CHECK-LABEL: e5m2_e2m1_m16n16k128
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i32(<8 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x float> zeroinitializer, i32 1, i32 4, i32 0, i32 0, i32 0, i32 0)
TEST(e5m2_e2m1_m16n16k128, fp8_e5m2, fp4_e2m1, float, 16, 16, 128)
