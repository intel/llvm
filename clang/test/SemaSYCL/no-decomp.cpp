// RUN: %clang_cc1 -fsycl-is-device -triple nvptx64-nvidia-cuda -ast-dump %s | FileCheck %s -check-prefix=ALL,NODECOMP
// RUN: %clang_cc1 -fsycl-is-device -fno-sycl-decompose-functor -triple nvptx64-nvidia-cuda -ast-dump %s | FileCheck %s -check-prefix=ALL,NODECOMP
// RUN: %clang_cc1 -fsycl-is-device -fsycl-decompose-functor -triple nvptx64-nvidia-cuda -ast-dump %s | FileCheck %s -check-prefix=ALL,DECOMP

#include "Inputs/sycl.hpp"

class with_acc {
public:
  int *d;
  sycl::accessor<char, 1, sycl::access::mode::read> AccField;
};

class wraping_acc {
public:
  with_acc acc;
  void operator()() const {
  }
};

class pointer_wrap {
public:
  int *d;
  void operator()() const {
  }
};

class empty {
public:
  void operator()() const {
  }
};

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    wraping_acc acc;
    cgh.single_task(acc);
  });
  // ALL: FunctionDecl {{.*}} _ZTS11wraping_acc 'void (__wrapper_class, __global char *, sycl::range<1>, sycl::range<1>, sycl::id<1>)'

  q.submit([&](sycl::handler &cgh) {
    pointer_wrap ptr;
    cgh.single_task(ptr);
  });
  // NODECOMP: FunctionDecl {{.*}} _ZTS12pointer_wrap 'void (pointer_wrap)'
  // DECOMP: FunctionDecl {{.*}} _ZTS12pointer_wrap 'void (__global int *)'

  q.submit([&](sycl::handler &cgh) {
    empty e;
    cgh.single_task(e);
  });
  // ALL: FunctionDecl {{.*}} _ZTS5empty 'void ()'

  return 0;
}
