// RUN: %clang_cc1 -fsycl -fsycl-is-device -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test checks if the compiler generates correct kernel wrapper arguments for
// image accessors targets.

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

queue q;

int main() {

  // 1-dimensional accessor with Read-only access
  accessor<int, 1, access::mode::read,
           access::target::image, access::placeholder::false_t>
      image_acc1d_read;

  q.submit([&](handler &h) {
    h.single_task<class use_image1d_r>(
        [=]() {
          image_acc1d_read.use();
        });
  });

  // 2-dimensional accessor with Read-only access
  accessor<int, 2, access::mode::read,
           access::target::image, access::placeholder::false_t>
      image_acc2d_read;
  q.submit([&](handler &h) {
    h.single_task<class use_image2d_r>(
        [=]() {
          image_acc2d_read.use();
        });
  });

  // 3-dimensional accessor with Read-only access
  accessor<int, 3, access::mode::read,
           access::target::image, access::placeholder::false_t>
      image_acc3d_read;

  q.submit([&](handler &h) {
    h.single_task<class use_image3d_r>(
        [=]() {
          image_acc3d_read.use();
        });
  });

  // 1-dimensional accessor with Write-only access
  accessor<int, 1, access::mode::write,
           access::target::image, access::placeholder::false_t>
      image_acc1d_write;
  q.submit([&](handler &h) {
    h.single_task<class use_image1d_w>(
        [=]() {
          image_acc1d_write.use();
        });
  });

  // 2-dimensional accessor with Write-only access
  accessor<int, 2, access::mode::write,
           access::target::image, access::placeholder::false_t>
      image_acc2d_write;
  q.submit([&](handler &h) {
    h.single_task<class use_image2d_w>(
        [=]() {
          image_acc2d_write.use();
        });
  });

  // 3-dimensional accessor with Write-only access
  accessor<int, 3, access::mode::write,
           access::target::image, access::placeholder::false_t>
      image_acc3d_write;
  q.submit([&](handler &h) {
    h.single_task<class use_image3d_w>(
        [=]() {
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
