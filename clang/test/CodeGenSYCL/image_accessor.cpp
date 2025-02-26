// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-1DRO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-2DRO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-3DRO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-1DWO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-2DWO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-3DWO
//
// CHECK-1DRO: define {{.*}}spir_kernel void @{{.*}}(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-1DRO: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{.*}} %{{[a-zA-Z0-9_]+}}, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %{{[0-9]+}})
//
// CHECK-2DRO: define {{.*}}spir_kernel void @{{.*}}(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-2DRO: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{.*}} %{{[a-zA-Z0-9_]+}}, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %{{[0-9]+}})
//
// CHECK-3DRO: define {{.*}}spir_kernel void @{{.*}}(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-3DRO: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{.*}} %{{[a-zA-Z0-9_]+}}, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %{{[0-9]+}})
//
// CHECK-1DWO: define {{.*}}spir_kernel void @{{.*}}(target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-1DWO: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{.*}} %{{[a-zA-Z0-9_]+}}, target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) %{{[0-9]+}})
//
// CHECK-2DWO: define {{.*}}spir_kernel void @{{.*}}(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-2DWO: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{.*}} %{{[a-zA-Z0-9_]+}}, target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %{{[0-9]+}})
//
// CHECK-3DWO: define {{.*}}spir_kernel void @{{.*}}(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 1) [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-3DWO: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{.*}} %{{[a-zA-Z0-9_]+}}, target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 1) %{{[0-9]+}})
//
// TODO: Add tests for the image_array opencl datatype support.
#include "Inputs/sycl.hpp"

int main() {

  {
    sycl::image<1> MyImage1d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<1>(3));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage1d.get_access<int, sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor1dro>([=]() {
        Acc.use();
      });
    });
  }

  {
    sycl::image<2> MyImage2d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<2>(3, 2));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage2d.get_access<int, sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor2dro>([=]() {
        Acc.use();
      });
    });
  }

  {
    sycl::image<3> MyImage3d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<3>(3, 2, 4));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage3d.get_access<int, sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor3dro>([=]() {
        Acc.use();
      });
    });
  }

  {
    sycl::image<1> MyImage1d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<1>(3));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage1d.get_access<int, sycl::access::mode::write>(cgh);

      cgh.single_task<class image_accessor1dwo>([=]() {
        Acc.use();
      });
    });
  }

  {
    sycl::image<2> MyImage2d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<2>(3, 2));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage2d.get_access<int, sycl::access::mode::write>(cgh);

      cgh.single_task<class image_accessor2dwo>([=]() {
        Acc.use();
      });
    });
  }

  {
    sycl::image<3> MyImage3d(sycl::image_channel_order::rgbx, sycl::image_channel_type::unorm_short_565, sycl::range<3>(3, 2, 4));
    sycl::queue Q;
    Q.submit([&](sycl::handler &cgh) {
      auto Acc = MyImage3d.get_access<int, sycl::access::mode::write>(cgh);

      cgh.single_task<class image_accessor3dwo>([=]() {
        Acc.use();
      });
    });
  }

  return 0;
}
