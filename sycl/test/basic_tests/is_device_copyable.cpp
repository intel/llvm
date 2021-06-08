// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify \
// RUN: %s -I %sycl_include 2>&1 | FileCheck -check-prefix=CHECK-ERR %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify \
// RUN: %s -DCOPYABLE -I %sycl_include

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

struct C : A {
  const A C2;
  C() : A{0}, C2{2}{}
};

struct D {
  int i;
  ~D();
};

#if defined(COPYABLE) && (SYCL_DEVICE_COPYABLE == 1)
template<> struct is_device_copyable<B> : std::true_type {};
template<> struct is_device_copyable<D> : std::true_type {};
// expected-no-diagnostics
#endif

void test() {
  A IamGood;
  IamGood.i = 0;
  B IamBad(1);
  C IamAlsoGood;
  D IamAlsoBad{0};
  sycl::queue Q;
  Q.single_task<class kernel_capture_refs>([=] {
    int a = IamGood.i;
    int b = IamBad.i;
    int c = IamAlsoGood.i;
    int d = IamAlsoBad.i;
  });
}
// CHECK-ERR: no expected directives found
// CHECK-ERR-NEXT: diagnostics seen but not expected
// CHECK-ERR-NEXT: is_device_copyable<B, void>
// CHECK-ERR-NEXT: is_device_copyable<D, void>

// CHECK-DEF: no expected directives found
