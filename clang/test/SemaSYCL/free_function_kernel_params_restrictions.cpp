// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -verify %s
// This test checks that compiler correctly diagnoses violations of restrictions
// applied to free function kernel parameters defined by the spec.

#include "sycl.hpp"

class Outer {
public:
  class DefinedWithinAClass {
    int f;
  };
};

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_4(Outer::DefinedWithinAClass S1) { // expected-error {{'Outer::DefinedWithinAClass' cannot be used as the type of a kernel parameter}}
                                           // expected-note@-1 {{'Outer::DefinedWithinAClass' is not forward declarable}}
}

template <typename T1>
__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
  void ff_6(T1 S1) { // expected-error 2{{cannot be used as the type of a kernel parameter}}
                     // expected-note@-1 2{{is not forward declarable}}
}

void bar() {
  ff_6([=](){});
}

auto Glob = [](int P){ return P + 1;};

template void ff_6(typeof(Glob) S1);

extern "C" {
  struct A {
    int a;
  };
}

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_5(A S1) {
}



struct StructWithAccessor {
  sycl::accessor<char, 1, sycl::access::mode::read> acc;
  int *ptr;
};

struct Wrapper {
  StructWithAccessor SWA;

};

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_6(Wrapper S1) { // expected-error {{cannot be used as the type of a kernel parameter}}
                        // expected-note@-1 {{'Wrapper' is not yet supported as a free function kernel parameter}}
}
