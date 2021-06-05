// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fcxx-exceptions -sycl-std=2020 -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// an array of non-trivially copyable structs as SYCL kernel parameter or
// a non-constant size array.

#include "sycl.hpp"

sycl::queue q;

struct NonTrivialCopyStruct {
  int i;
  NonTrivialCopyStruct(int _i) : i(_i) {}
  NonTrivialCopyStruct(const NonTrivialCopyStruct &x) : i(x.i) {}
};

struct NonTrivialDestructorStruct {
  int i;
  ~NonTrivialDestructorStruct();
};

class Array {
  // expected-error@+1 {{kernel parameter is not a constant size array}}
  int NonConstantSizeArray[];

public:
  int operator()() const { return NonConstantSizeArray[0]; }
};

void test() {
  NonTrivialCopyStruct NTCSObject[4] = {1, 2, 3, 4};
  NonTrivialDestructorStruct NTDSObject[5];
  // expected-note@+1 {{'UnknownSizeArrayObj' declared here}}
  Array UnknownSizeArrayObj;

  q.submit([&](sycl::handler &h) {
    h.single_task<class kernel_capture_refs>([=] {
      // expected-error@+1 {{kernel parameter has non-trivially copy constructible class/struct type}}
      int b = NTCSObject[2].i;
      // expected-error@+1 {{kernel parameter has non-trivially destructible class/struct type}}
      int d = NTDSObject[4].i;
    });
  });

  q.submit([&](sycl::handler &h) {
    // expected-error@+1 {{variable 'UnknownSizeArrayObj' with flexible array member cannot be captured in a lambda expression}}
    h.single_task<class kernel_bad_array>(UnknownSizeArrayObj);
  });
}
