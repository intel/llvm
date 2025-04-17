// RUN: %clang_cc1 %s -fsycl-is-host -fsycl-cuda-compatibility \
// RUN:   -internal-isystem %S/../../SemaCUDA/Inputs \
// RUN:   -internal-isystem %S/../Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux \
// RUN:   -o %t -verify
// RUN: %clang_cc1 %s -fsycl-is-device -fsycl-cuda-compatibility \
// RUN:   -internal-isystem %S/../../SemaCUDA/Inputs \
// RUN:   -internal-isystem %S/../Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux\
// RUN:   -o %t -verify

// Check that delayed diagnostics are done
// according to the SYCL logic and not the CUDA one.
// fsycl-cuda-compatibility doesn't enable the CUDA device mode,
// this leaves the CUDA diagnostic logic think it is processing the host side
// and triggers SYCL's delayed diagnostics.

// expected-no-diagnostics

#include "cuda.h"
#include "sycl.hpp"

struct Foo {
  Foo();
};

Foo& get() {
  // Checks no error is raised for non const static variable.
  static Foo f;
  return f;
}

extern "C" __attribute__((device)) int printf(const char *fmt, ...);

void print() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image1d_r>(
        [=] {
          printf("hello");
        });
  });
}
