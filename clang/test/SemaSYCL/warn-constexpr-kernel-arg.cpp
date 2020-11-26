// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -Wsycl-strict -sycl-std=2020 -verify %s

#include "sycl.hpp"

using namespace cl::sycl;

queue q;

class LambdaKernel;

int main() {

  constexpr unsigned V = 16;
  struct Id {
    int x;

    Id(int x) : x(x) {}

    Id operator*(const unsigned &val) const {
      return Id(x * val);
    }

    operator int() const {
      return x;
    }
  };

  Id i{5};

  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<LambdaKernel>(
        [=]() {
          int A = i * V; // expected-warning {{constexpr is captured and is not a compile-time constant}}
        });
  });
  return 0;
}
