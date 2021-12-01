// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl-device-only -fsycl-unnamed-lambda -Xclang -disable-llvm-passes %s -o - | FileCheck %s

#include <CL/sycl.hpp>

int main() {
  using namespace sycl;
  queue q;
  q.single_task([] {
    half h(0.);
    half2 h2(0., 0.);
    half4 h4(0., 0., 0., 0.);
    half8 h8(0., 0., 0., 0., 0., 0., 0., 0.);
    half16 h16(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);

    float f(0.);
    float2 f2(0., 0.);
    float4 f4(0., 0., 0., 0.);
    float8 f8(0., 0., 0., 0., 0., 0., 0., 0.);
    float16 f16(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);

    double d(0.);
    double2 d2(0., 0.);
    double4 d4(0., 0., 0., 0.);
    double8 d8(0., 0., 0., 0., 0., 0., 0., 0.);
    double16 d16(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0.);

    {
      // CHECK: call spir_func zeroext i1 @_Z13__spirv_IsNanDF16_(
      (void)isnan(h);
      // CHECK: call spir_func <2 x i16> @_Z13__spirv_IsNanDv2_DF16_(
      (void)isnan(h2);
      // CHECK: call spir_func <4 x i16> @_Z13__spirv_IsNanDv4_DF16_(
      (void)isnan(h4);
      // CHECK: call spir_func <8 x i16> @_Z13__spirv_IsNanDv8_DF16_(
      (void)isnan(h8);
      // CHECK: call spir_func <16 x i16> @_Z13__spirv_IsNanDv16_DF16_(
      (void)isnan(h16);

      // CHECK: call spir_func zeroext i1 @_Z13__spirv_IsNanf(
      (void)isnan(f);
      // CHECK: call spir_func <2 x i32> @_Z13__spirv_IsNanDv2_f(
      (void)isnan(f2);
      // CHECK: call spir_func <4 x i32> @_Z13__spirv_IsNanDv4_f(
      (void)isnan(f4);
      // CHECK: call spir_func <8 x i32> @_Z13__spirv_IsNanDv8_f(
      (void)isnan(f8);
      // CHECK: call spir_func <16 x i32> @_Z13__spirv_IsNanDv16_f(
      (void)isnan(f16);

      // CHECK: call spir_func zeroext i1 @_Z13__spirv_IsNand(
      (void)isnan(d);
      // CHECK: call spir_func <2 x i64> @_Z13__spirv_IsNanDv2_d(
      (void)isnan(d2);
      // CHECK: call spir_func <4 x i64> @_Z13__spirv_IsNanDv4_d(
      (void)isnan(d4);
      // CHECK: call spir_func <8 x i64> @_Z13__spirv_IsNanDv8_d(
      (void)isnan(d8);
      // CHECK: call spir_func <16 x i64> @_Z13__spirv_IsNanDv16_d(
      (void)isnan(d16);
    }

    {
      // CHECK: call spir_func zeroext i1 @_Z21__spirv_LessOrGreaterDF16_DF16_(
      (void)islessgreater(h, h);
      // CHECK: call spir_func <2 x i16> @_Z21__spirv_LessOrGreaterDv2_DF16_S_(
      (void)islessgreater(h2, h2);
      // CHECK: call spir_func <4 x i16> @_Z21__spirv_LessOrGreaterDv4_DF16_S_(
      (void)islessgreater(h4, h4);
      // CHECK: call spir_func <8 x i16> @_Z21__spirv_LessOrGreaterDv8_DF16_S_(
      (void)islessgreater(h8, h8);
      // CHECK: call spir_func <16 x i16> @_Z21__spirv_LessOrGreaterDv16_DF16_S_(
      (void)islessgreater(h16, h16);

      // CHECK: call spir_func zeroext i1 @_Z21__spirv_LessOrGreaterff(
      (void)islessgreater(f, f);
      // CHECK: call spir_func <2 x i32> @_Z21__spirv_LessOrGreaterDv2_fS_(
      (void)islessgreater(f2, f2);
      // CHECK: call spir_func <4 x i32> @_Z21__spirv_LessOrGreaterDv4_fS_(
      (void)islessgreater(f4, f4);
      // CHECK: call spir_func <8 x i32> @_Z21__spirv_LessOrGreaterDv8_fS_(
      (void)islessgreater(f8, f8);
      // CHECK: call spir_func <16 x i32> @_Z21__spirv_LessOrGreaterDv16_fS_(
      (void)islessgreater(f16, f16);

      // CHECK: call spir_func zeroext i1 @_Z21__spirv_LessOrGreaterdd(
      (void)islessgreater(d, d);
      // CHECK: call spir_func <2 x i64> @_Z21__spirv_LessOrGreaterDv2_dS_(
      (void)islessgreater(d2, d2);
      // CHECK: call spir_func <4 x i64> @_Z21__spirv_LessOrGreaterDv4_dS_(
      (void)islessgreater(d4, d4);
      // CHECK: call spir_func <8 x i64> @_Z21__spirv_LessOrGreaterDv8_dS_(
      (void)islessgreater(d8, d8);
      // CHECK: call spir_func <16 x i64> @_Z21__spirv_LessOrGreaterDv16_dS_(
      (void)islessgreater(d16, d16);
    }
  });

  return 0;
}
