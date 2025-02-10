// RUN: %clang_cc1 %s -fsycl-is-host -fsycl-cuda-compatibility \
// RUN:   -internal-isystem %S/../../SemaCUDA/Inputs \
// RUN:   -internal-isystem %S/../Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux \
// RUN:   -emit-llvm -o - -verify -verify-ignore-unexpected=note
// RUN: %clang_cc1 %s -fsycl-is-device -fsycl-cuda-compatibility \
// RUN:   -internal-isystem %S/../../SemaCUDA/Inputs \
// RUN:   -internal-isystem %S/../Inputs \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda -triple x86_64-unknown-linux\
// RUN:   -emit-llvm -o - -verify -verify-ignore-unexpected=note


// expected-no-diagnostics

#include "cuda.h"
#include "sycl.hpp"

struct Foo {
  Foo();
};

Foo& get() {
  static Foo f;
  return f;
}

int printf(const char *restrict, ...) { return 0; }

void print() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task<class use_image1d_r>(
        [=] {
          printf("hello");
        });
  });
}
