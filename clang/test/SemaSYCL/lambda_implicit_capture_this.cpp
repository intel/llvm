// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -verify %s

#include <sycl.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

class Class {
public:
  Class() : member(1) {}
  void function();
  int member;
};

void Class::function() {
  kernel<class kernel_wrapper>(
      [=]() {
        int acc[1] = {5};
        acc[0] *= member; // expected-error{{implicit capture of 'this' with a capture default of '=' is not allowed for kernel functions in SYCL 1.2.1}}
      });
}

int main(int argc, char *argv[]) {
  Class c;
  c.function();
}

