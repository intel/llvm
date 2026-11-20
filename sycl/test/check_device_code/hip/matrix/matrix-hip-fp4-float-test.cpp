// REQUIRES: hip
// REQUIRES: hip-arch-gfx950
// RUN: %clangxx -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -S %s -o -| FileCheck %s

// OCP FP4 multiplicands (packed fp4_e2m1_x2), alone or mixed with FP8, use the
// F8F6F4 MFMAs (16x16x128, 32x32x64), whose CBSZ/BLGP operands select the A/B
// formats (0 = E4M3, 1 = E5M2, 4 = E2M1). An FP4 operand is 128 bits wide.

#include <sycl/ext/oneapi/experimental/float_4bit/types.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::experimental::fp4_e2m1_x2;
using sycl::ext::oneapi::experimental::fp8_e4m3;
using sycl::ext::oneapi::experimental::fp8_e5m2;

// Number of matrix values held by one element of T.
template <typename T> constexpr size_t values_per_element() {
  return std::is_same_v<T, fp4_e2m1_x2> ? 2 : 1;
}

template <typename TA, typename TB, size_t M, size_t N, size_t K>
void mad(TA *A, TB *B, float *D, sub_group sg) {
  joint_matrix<sub_group, float, use::accumulator, M, N> sub_c{};
  joint_matrix<sub_group, TA, use::a, M, K, layout::row_major> sub_a{};
  joint_matrix<sub_group, TB, use::b, K, N, layout::col_major> sub_b{};
  joint_matrix_load(sg, sub_a,
                    address_space_cast<access::address_space::global_space,
                                       access::decorated::no>(A),
                    K / values_per_element<TA>());
  joint_matrix_load(sg, sub_b,
                    address_space_cast<access::address_space::global_space,
                                       access::decorated::no>(B),
                    K / values_per_element<TB>());
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

// CHECK-LABEL: e2m1_e2m1_m16n16k128
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
TEST(e2m1_e2m1_m16n16k128, fp4_e2m1_x2, fp4_e2m1_x2, 16, 16, 128)
// CHECK-LABEL: e2m1_e2m1_m32n32k64
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v4i32(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <16 x float> zeroinitializer, i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)
TEST(e2m1_e2m1_m32n32k64, fp4_e2m1_x2, fp4_e2m1_x2, 32, 32, 64)
// CHECK-LABEL: e2m1_e4m3_m32n32k64
// CHECK: tail call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v4i32.v8i32(<4 x i32> {{.*}}, <8 x i32> {{.*}}, <16 x float> zeroinitializer, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
TEST(e2m1_e4m3_m32n32k64, fp4_e2m1_x2, fp8_e4m3, 32, 32, 64)
// CHECK-LABEL: e5m2_e2m1_m16n16k128
// CHECK: tail call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v4i32(<8 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x float> zeroinitializer, i32 1, i32 4, i32 0, i32 0, i32 0, i32 0)
TEST(e5m2_e2m1_m16n16k128, fp8_e5m2, fp4_e2m1_x2, 16, 16, 128)
