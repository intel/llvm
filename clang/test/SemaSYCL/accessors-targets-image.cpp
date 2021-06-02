// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks if the compiler generates correct kernel wrapper arguments for
// image accessors targets.

#include "sycl.hpp"

sycl::queue q;

int main() {

  sycl::accessor<int, 1, sycl::access::mode::read,
                 sycl::access::target::image, sycl::access::placeholder::false_t>
      image_acc1d_read;

  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image1d_r>(
        [=] {
          image_acc1d_read.use();
        });
  });

  sycl::accessor<int, 2, sycl::access::mode::read,
                 sycl::access::target::image, sycl::access::placeholder::false_t>
      image_acc2d_read;
  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image2d_r>(
        [=] {
          image_acc2d_read.use();
        });
  });

  sycl::accessor<int, 3, sycl::access::mode::read,
                 sycl::access::target::image, sycl::access::placeholder::false_t>
      image_acc3d_read;

  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image3d_r>(
        [=] {
          image_acc3d_read.use();
        });
  });

  sycl::accessor<int, 1, sycl::access::mode::write,
                 sycl::access::target::image, sycl::access::placeholder::false_t>
      image_acc1d_write;
  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image1d_w>(
        [=] {
          image_acc1d_write.use();
        });
  });

  sycl::accessor<int, 2, sycl::access::mode::write,
                 sycl::access::target::image, sycl::access::placeholder::false_t>
      image_acc2d_write;
  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image2d_w>(
        [=] {
          image_acc2d_write.use();
        });
  });

  sycl::accessor<int, 3, sycl::access::mode::write,
                 sycl::access::target::image, sycl::access::placeholder::false_t>
      image_acc3d_write;
  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image3d_w>(
        [=] {
          image_acc3d_write.use();
        });
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
