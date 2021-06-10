// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify \
// RUN: %s -I %sycl_include 2>&1 | FileCheck -check-prefix=CHECK-ERR %s

// This test checks if compiler reports compilation error on an attempt to pass
// a struct with type that is not device copyable as SYCL kernel parameter.
// It also checks that the error is not emitted if the type trait
// is_device_copyable is specialized for such types as true_type.

#include <CL/sycl.hpp>

using namespace cl::sycl;

struct A { int i; };

struct B {
  int i;
  B (int _i) : i(_i) {}
  B (const B& x) : i(x.i) {}
};

// Identical to B, but will be declared as device copyable.
struct BCopyable {
  int i;
  BCopyable (int _i) : i(_i) {}
  BCopyable (const BCopyable& x) : i(x.i) {}
};

struct C : A {
  const A C2;
  C() : A{0}, C2{2}{}
};

struct D {
  int i;
  ~D();
};

// Identical to D, but will be declared as device copyable.
struct DCopyable {
  int i;
  ~DCopyable();
};

template<> struct is_device_copyable<BCopyable> : std::true_type {};
template<> struct is_device_copyable<DCopyable> : std::true_type {};

void test() {
  A IamGood;
  IamGood.i = 0;
  B IamBad(1);
  BCopyable IamBadButCopyable(1);
  C IamAlsoGood;
  D IamAlsoBad{0};
  DCopyable IamAlsoBadButCopyable{0};
  queue Q;
  Q.single_task<class kernel_capture_refs>([=] {
    int A = IamGood.i;
    int B = IamBad.i;

    int C = IamBadButCopyable.i;
    int D = IamAlsoBadButCopyable.i;

    int E = IamAlsoGood.i;
    int F = IamAlsoBad.i;
  });
}
// CHECK-ERR: no expected directives found
// CHECK-ERR-NEXT: diagnostics seen but not expected
// CHECK-ERR-NEXT: is_device_copyable<B, void>
// CHECK-ERR-NEXT: is_device_copyable<D, void>
