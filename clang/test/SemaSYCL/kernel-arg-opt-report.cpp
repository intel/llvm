// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device \
// RUN: -Wno-sycl-2017-compat -emit-llvm-bc %s -o %t-host.bc -opt-record-file %t-host.yaml

#include "Inputs/sycl.hpp"

class second_base {
public:
  int *e;
};

class InnerFieldBase {
public:
  int d;
};
class InnerField : public InnerFieldBase {
  int c;
};

struct base {
public:
  int b;
  InnerField obj;
};

struct derived : base, second_base {
  int a;

  void operator()() const {
  }
};

int main() {
  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}

