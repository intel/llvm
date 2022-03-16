// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-1DRO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-2DRO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-3DRO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-1DWO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-2DWO
// RUN: FileCheck < %t.ll --enable-var-scope %s --check-prefix=CHECK-3DWO
//
// CHECK-1DRO: %opencl.image1d_ro_t = type opaque
// CHECK-1DRO: define {{.*}}spir_kernel void @{{.*}}(%opencl.image1d_ro_t addrspace(1)* [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-1DRO: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::accessor{{.*}} %{{[a-zA-Z]+}}, %opencl.image1d_ro_t addrspace(1)* %{{[0-9]+}})
//
// CHECK-2DRO: %opencl.image2d_ro_t = type opaque
// CHECK-2DRO: define {{.*}}spir_kernel void @{{.*}}(%opencl.image2d_ro_t addrspace(1)* [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-2DRO: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::accessor{{.*}} %{{[a-zA-Z]+}}, %opencl.image2d_ro_t addrspace(1)* %{{[0-9]+}})
//
// CHECK-3DRO: %opencl.image3d_ro_t = type opaque
// CHECK-3DRO: define {{.*}}spir_kernel void @{{.*}}(%opencl.image3d_ro_t addrspace(1)* [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-3DRO: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::accessor{{.*}} %{{[a-zA-Z]+}}, %opencl.image3d_ro_t addrspace(1)* %{{[0-9]+}})
//
// CHECK-1DWO: %opencl.image1d_wo_t = type opaque
// CHECK-1DWO: define {{.*}}spir_kernel void @{{.*}}(%opencl.image1d_wo_t addrspace(1)* [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-1DWO: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::accessor{{.*}} %{{[a-zA-Z]+}}, %opencl.image1d_wo_t addrspace(1)* %{{[0-9]+}})
//
// CHECK-2DWO: %opencl.image2d_wo_t = type opaque
// CHECK-2DWO: define {{.*}}spir_kernel void @{{.*}}(%opencl.image2d_wo_t addrspace(1)* [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-2DWO: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::accessor{{.*}} %{{[a-zA-Z]+}}, %opencl.image2d_wo_t addrspace(1)* %{{[0-9]+}})
//
// CHECK-3DWO: %opencl.image3d_wo_t = type opaque
// CHECK-3DWO: define {{.*}}spir_kernel void @{{.*}}(%opencl.image3d_wo_t addrspace(1)* [[IMAGE_ARG:%[a-zA-Z0-9_]+]])
// CHECK-3DWO: call spir_func void @{{.*}}__init{{.*}}(%{{.*}}cl::sycl::accessor{{.*}} %{{[a-zA-Z]+}}, %opencl.image3d_wo_t addrspace(1)* %{{[0-9]+}})
//
// TODO: Add tests for the image_array opencl datatype support.
#include "Inputs/sycl.hpp"

int main() {

  {
    cl::sycl::image<1> MyImage1d(cl::sycl::image_channel_order::rgbx, cl::sycl::image_channel_type::unorm_short_565, cl::sycl::range<1>(3));
    cl::sycl::queue Q;
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = MyImage1d.get_access<int, cl::sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor1dro>([=]() {
        Acc.use();
      });
    });
  }

  {
    cl::sycl::image<2> MyImage2d(cl::sycl::image_channel_order::rgbx, cl::sycl::image_channel_type::unorm_short_565, cl::sycl::range<2>(3, 2));
    cl::sycl::queue Q;
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = MyImage2d.get_access<int, cl::sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor2dro>([=]() {
        Acc.use();
      });
    });
  }

  {
    cl::sycl::image<3> MyImage3d(cl::sycl::image_channel_order::rgbx, cl::sycl::image_channel_type::unorm_short_565, cl::sycl::range<3>(3, 2, 4));
    cl::sycl::queue Q;
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = MyImage3d.get_access<int, cl::sycl::access::mode::read>(cgh);

      cgh.single_task<class image_accessor3dro>([=]() {
        Acc.use();
      });
    });
  }

  {
    cl::sycl::image<1> MyImage1d(cl::sycl::image_channel_order::rgbx, cl::sycl::image_channel_type::unorm_short_565, cl::sycl::range<1>(3));
    cl::sycl::queue Q;
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = MyImage1d.get_access<int, cl::sycl::access::mode::write>(cgh);

      cgh.single_task<class image_accessor1dwo>([=]() {
        Acc.use();
      });
    });
  }

  {
    cl::sycl::image<2> MyImage2d(cl::sycl::image_channel_order::rgbx, cl::sycl::image_channel_type::unorm_short_565, cl::sycl::range<2>(3, 2));
    cl::sycl::queue Q;
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = MyImage2d.get_access<int, cl::sycl::access::mode::write>(cgh);

      cgh.single_task<class image_accessor2dwo>([=]() {
        Acc.use();
      });
    });
  }

  {
    cl::sycl::image<3> MyImage3d(cl::sycl::image_channel_order::rgbx, cl::sycl::image_channel_type::unorm_short_565, cl::sycl::range<3>(3, 2, 4));
    cl::sycl::queue Q;
    Q.submit([&](cl::sycl::handler &cgh) {
      auto Acc = MyImage3d.get_access<int, cl::sycl::access::mode::write>(cgh);

      cgh.single_task<class image_accessor3dwo>([=]() {
        Acc.use();
      });
    });
  }

  return 0;
}
