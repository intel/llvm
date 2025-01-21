// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-X86
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -O0 -emit-llvm -o - -cl-std=clc++ | FileCheck %s --check-prefix=CHECK-X86
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -O0 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SPIR
// RUN: %clang_cc1 %s -triple spir64-unknown-unknown -O0 -emit-llvm -o - -cl-std=clc++ | FileCheck %s --check-prefix=CHECK-SPIR

__attribute__((overloadable)) void my_read_image(__ocl_sampled_image1d_ro_t img);
__attribute__((overloadable)) void my_read_image(__ocl_sampled_image2d_ro_t img);

void test_read_image(__ocl_sampled_image1d_ro_t img_ro, __ocl_sampled_image2d_ro_t img_2d) {
  // CHECK-X86: call void @_Z13my_read_image32__spirv_SampledImage__image1d_ro(ptr %{{[0-9]+}})
  // CHECK-SPIR: call spir_func void @_Z13my_read_image32__spirv_SampledImage__image1d_ro(target("spirv.SampledImage", void, 0, 0, 0, 0, 0, 0, 0) %{{[0-9]+}})
  my_read_image(img_ro);
  // CHECK-X86: call void @_Z13my_read_image32__spirv_SampledImage__image2d_ro(ptr %{{[0-9]+}})
  // CHECK-SPIR: call spir_func void @_Z13my_read_image32__spirv_SampledImage__image2d_ro(target("spirv.SampledImage", void, 1, 0, 0, 0, 0, 0, 0) %{{[0-9]+}})
  my_read_image(img_2d);
}
