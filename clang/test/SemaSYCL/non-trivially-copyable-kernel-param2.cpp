// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -Wno-sycl-2017-compat -fsyntax-only -fsycl-int-footer=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks if the device compiler pass creates an integration footer
// which tests for kernel parameter copyability.

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

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

void test() {
  A IamGood;
  IamGood.i = 0;
  B IamBad(1);
  C IamAlsoGood;
  D IamAlsoBad{0};
  kernel_single_task<class kernel_capture_refs>([=] {
    int a = IamGood.i;
    int b = IamBad.i;
    int c = IamAlsoGood.i;
    int d = IamAlsoBad.i;
  });
}

// CHECK-LABEL: #include <CL/sycl/detail/sycl_fe_intrins.hpp>
// CHECK: static_assert(::sycl::is_device_copyable<A>_v, "error: kernel parameter type ('A') is not device copyable");
// CHECK-NOT: static_assert(::sycl::is_device_copyable<const A>_v, "error: kernel parameter type ('A') is not device copyable");
// CHECK-NEXT: static_assert(::sycl::is_device_copyable<B>_v, "error: kernel parameter type ('B') is not device copyable");
// CHECK-NEXT: static_assert(::sycl::is_device_copyable<C>_v, "error: kernel parameter type ('C') is not device copyable");
// CHECK-NEXT: static_assert(::sycl::is_device_copyable<D>_v, "error: kernel parameter type ('D') is not device copyable");
