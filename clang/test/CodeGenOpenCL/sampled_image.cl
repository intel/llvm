// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - -cl-std=clc++ | FileCheck %s

__attribute__((overloadable)) void my_read_image(__ocl_sampled_image1d_ro_t img);
__attribute__((overloadable)) void my_read_image(__ocl_sampled_image2d_ro_t img);

void test_read_image(__ocl_sampled_image1d_ro_t img_ro, __ocl_sampled_image2d_ro_t img_2d) {
  // CHECK: call void @_Z13my_read_image32__spirv_SampledImage__image1d_ro(ptr %{{[0-9]+}})
  my_read_image(img_ro);
  // CHECK: call void @_Z13my_read_image32__spirv_SampledImage__image2d_ro(ptr %{{[0-9]+}})
  my_read_image(img_2d);
}
