// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper arguments for
// image accessors targets.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  accessor<int, 1, access::mode::read,
           access::target::image, access::placeholder::false_t>
      image_acc1d_read;
  kernel<class use_image1d_r>(
      [=]() {
        image_acc1d_read.use();
      });

  accessor<int, 2, access::mode::read,
           access::target::image, access::placeholder::false_t>
      image_acc2d_read;
  kernel<class use_image2d_r>(
      [=]() {
        image_acc2d_read.use();
      });

  accessor<int, 3, access::mode::read,
           access::target::image, access::placeholder::false_t>
      image_acc3d_read;
  kernel<class use_image3d_r>(
      [=]() {
        image_acc3d_read.use();
      });

  accessor<int, 1, access::mode::write,
           access::target::image, access::placeholder::false_t>
      image_acc1d_write;
  kernel<class use_image1d_w>(
      [=]() {
        image_acc1d_write.use();
      });

  accessor<int, 2, access::mode::write,
           access::target::image, access::placeholder::false_t>
      image_acc2d_write;
  kernel<class use_image2d_w>(
      [=]() {
        image_acc2d_write.use();
      });

  accessor<int, 3, access::mode::write,
           access::target::image, access::placeholder::false_t>
      image_acc3d_write;
  kernel<class use_image3d_w>(
      [=]() {
        image_acc3d_write.use();
      });
}

// CHECK: {{.*}}use_image1d_r 'void (__read_only image1d_t)'
// CHECK: {{.*}}use_image2d_r 'void (__read_only image2d_t)'
// CHECK: {{.*}}use_image3d_r 'void (__read_only image3d_t)'
// CHECK: {{.*}}use_image1d_w 'void (__write_only image1d_t)'
// CHECK: {{.*}}use_image2d_w 'void (__write_only image2d_t)'
// CHECK: {{.*}}use_image3d_w 'void (__write_only image3d_t)'

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
