// RUN: %clang_cc1 %s -triple nvptx64-nvidia-cuda -O0 -emit-llvm -o - | FileCheck %s

__attribute__((overloadable)) void my_read_image(__ocl_sampled_image1d_ro_t img_ro);

__attribute__((overloadable)) __ocl_sampled_image1d_ro_t __spirv_SampledImage(read_only image1d_t img_wo, sampler_t sampl);

void test_read_image(__ocl_sampled_image1d_ro_t img_ro, read_only image1d_t img_wo, sampler_t sampl) {

  // CHECK: call void @_Z13my_read_image32__spirv_SampledImage__image1d_ro(i64 %{{[a-zA-Z0-9]+}}, i32 %{{[a-zA-Z0-9]+}})
  my_read_image(img_ro);
  // CHECK: call { i64, i32 } @_Z20__spirv_SampledImage14ocl_image1d_ro11ocl_sampler(i64 %{{[a-zA-Z0-9]+}}, i32 %{{[a-zA-Z0-9]+}})
  __ocl_sampled_image1d_ro_t s_imag = __spirv_SampledImage(img_wo, sampl);
  // CHECK: call void @_Z13my_read_image32__spirv_SampledImage__image1d_ro(i64 %{{[a-zA-Z0-9]+}}, i32 %{{[a-zA-Z0-9]+}})
  my_read_image(s_imag);
}
