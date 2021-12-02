// RUN: %clangxx -I %sycl_include -S -emit-llvm -fsycl-device-only -Xclang -disable-llvm-passes %s -o - | FileCheck %s
//
// Check relational builtin function type.

#include <CL/sycl.hpp>

int main() {
  sycl::queue q;
  q.single_task([] {
    sycl::half h(0.);
    sycl::half2 h2(0., 0.);
    sycl::half4 h4(0., 0., 0., 0.);
    sycl::half8 h8(0., 0., 0., 0., 0., 0., 0., 0.);
    sycl::half16 h16(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0.);

    float f(0.);
    sycl::float2 f2(0., 0.);
    sycl::float4 f4(0., 0., 0., 0.);
    sycl::float8 f8(0., 0., 0., 0., 0., 0., 0., 0.);
    sycl::float16 f16(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0.);

    double d(0.);
    sycl::double2 d2(0., 0.);
    sycl::double4 d4(0., 0., 0., 0.);
    sycl::double8 d8(0., 0., 0., 0., 0., 0., 0., 0.);
    sycl::double16 d16(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0., 0.);

    {
      // CHECK: call spir_func zeroext i1 @_Z13__spirv_IsNanDF16_(
      (void)sycl::isnan(h);
      // CHECK: call spir_func <2 x i16> @_Z13__spirv_IsNanDv2_DF16_(
      (void)sycl::isnan(h2);
      // CHECK: call spir_func <4 x i16> @_Z13__spirv_IsNanDv4_DF16_(
      (void)sycl::isnan(h4);
      // CHECK: call spir_func <8 x i16> @_Z13__spirv_IsNanDv8_DF16_(
      (void)sycl::isnan(h8);
      // CHECK: call spir_func <16 x i16> @_Z13__spirv_IsNanDv16_DF16_(
      (void)sycl::isnan(h16);

      // CHECK: call spir_func zeroext i1 @_Z13__spirv_IsNanf(
      (void)sycl::isnan(f);
      // CHECK: call spir_func <2 x i32> @_Z13__spirv_IsNanDv2_f(
      (void)sycl::isnan(f2);
      // CHECK: call spir_func <4 x i32> @_Z13__spirv_IsNanDv4_f(
      (void)sycl::isnan(f4);
      // CHECK: call spir_func <8 x i32> @_Z13__spirv_IsNanDv8_f(
      (void)sycl::isnan(f8);
      // CHECK: call spir_func <16 x i32> @_Z13__spirv_IsNanDv16_f(
      (void)sycl::isnan(f16);

      // CHECK: call spir_func zeroext i1 @_Z13__spirv_IsNand(
      (void)sycl::isnan(d);
      // CHECK: call spir_func <2 x i64> @_Z13__spirv_IsNanDv2_d(
      (void)sycl::isnan(d2);
      // CHECK: call spir_func <4 x i64> @_Z13__spirv_IsNanDv4_d(
      (void)sycl::isnan(d4);
      // CHECK: call spir_func <8 x i64> @_Z13__spirv_IsNanDv8_d(
      (void)sycl::isnan(d8);
      // CHECK: call spir_func <16 x i64> @_Z13__spirv_IsNanDv16_d(
      (void)sycl::isnan(d16);
    }

    {
      // CHECK: call spir_func zeroext i1 @_Z21__spirv_LessOrGreaterDF16_DF16_(
      (void)sycl::islessgreater(h, h);
      // CHECK: call spir_func <2 x i16> @_Z21__spirv_LessOrGreaterDv2_DF16_S_(
      (void)sycl::islessgreater(h2, h2);
      // CHECK: call spir_func <4 x i16> @_Z21__spirv_LessOrGreaterDv4_DF16_S_(
      (void)sycl::islessgreater(h4, h4);
      // CHECK: call spir_func <8 x i16> @_Z21__spirv_LessOrGreaterDv8_DF16_S_(
      (void)sycl::islessgreater(h8, h8);
      // CHECK: call spir_func <16 x i16> @_Z21__spirv_LessOrGreaterDv16_DF16_S_(
      (void)sycl::islessgreater(h16, h16);

      // CHECK: call spir_func zeroext i1 @_Z21__spirv_LessOrGreaterff(
      (void)sycl::islessgreater(f, f);
      // CHECK: call spir_func <2 x i32> @_Z21__spirv_LessOrGreaterDv2_fS_(
      (void)sycl::islessgreater(f2, f2);
      // CHECK: call spir_func <4 x i32> @_Z21__spirv_LessOrGreaterDv4_fS_(
      (void)sycl::islessgreater(f4, f4);
      // CHECK: call spir_func <8 x i32> @_Z21__spirv_LessOrGreaterDv8_fS_(
      (void)sycl::islessgreater(f8, f8);
      // CHECK: call spir_func <16 x i32> @_Z21__spirv_LessOrGreaterDv16_fS_(
      (void)sycl::islessgreater(f16, f16);

      // CHECK: call spir_func zeroext i1 @_Z21__spirv_LessOrGreaterdd(
      (void)sycl::islessgreater(d, d);
      // CHECK: call spir_func <2 x i64> @_Z21__spirv_LessOrGreaterDv2_dS_(
      (void)sycl::islessgreater(d2, d2);
      // CHECK: call spir_func <4 x i64> @_Z21__spirv_LessOrGreaterDv4_dS_(
      (void)sycl::islessgreater(d4, d4);
      // CHECK: call spir_func <8 x i64> @_Z21__spirv_LessOrGreaterDv8_dS_(
      (void)sycl::islessgreater(d8, d8);
      // CHECK: call spir_func <16 x i64> @_Z21__spirv_LessOrGreaterDv16_dS_(
      (void)sycl::islessgreater(d16, d16);
    }
  });

  return 0;
}
