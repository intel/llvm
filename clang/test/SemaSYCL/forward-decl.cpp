// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify -pedantic %s

// expected-no-diagnostics

#include "sycl.hpp"

sycl::queue deviceQueue;

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function>([]() {
      class Foo *F;
      class Boo {
      public:
        virtual int getBoo() { return 42; }
      };
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function_2>([]() {
      class Boo *B;
    });
  });

  return 0;
}
