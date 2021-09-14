// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only %s

#include "sycl.hpp"

sycl::queue deviceQueue;

typedef __float128 BIGTY;

template <class T>
class Z {
public:
  // expected-note@+2 {{'field' defined here}}
  // expected-error@+1 3{{'__float128' is not supported on this target}}
  T field;
  // expected-note@+2 2{{'field1' defined here}}
  // expected-error@+1 3{{'__float128' is not supported on this target}}
  __float128 field1;
  using BIGTYPE = __float128;
  // expected-note@+2 {{'bigfield' defined here}}
  // expected-error@+1 3{{'__float128' is not supported on this target}}
  BIGTYPE bigfield;
};

void host_ok(void) {
  __float128 A;
  int B = sizeof(__float128);
  Z<__float128> C;
  C.field1 = A;
}

void usage() {
  // expected-note@+2 3{{'A' defined here}}
  // expected-error@+1 {{'__float128' is not supported on this target}}
  __float128 A;
  // expected-note@+1 3{{used here}}
  Z<__float128> C;
  // expected-error@+2 {{'A' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
  // expected-error@+1 {{'field1' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
  C.field1 = A;
  // expected-error@+1 {{'bigfield' requires 128 bit size 'Z::BIGTYPE' (aka '__float128') type support, but device 'spir64' does not support it}}
  C.bigfield += 1.0;

  // expected-note@+2 {{used here}}
  // expected-error@+1 {{'A' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
  auto foo1 = [=]() {
    // expected-error@+1 {{'__float128' is not supported on this target}}
    __float128 AA;
    // expected-error@+3 2{{'__float128' is not supported on this target}}
    // expected-note@+2 {{'BB' defined here}}
    // expected-error@+1 {{'A' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
    auto BB = A;
    // expected-error@+1 {{'BB' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
    BB += 1;
  };

  // expected-note@+1 {{called by 'usage'}}
  foo1();
}

template <typename t>
void foo2(){};

// expected-note@+3 {{'P' defined here}}
// expected-error@+2 {{'P' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
// expected-note@+1 2{{'foo' defined here}}
__float128 foo(__float128 P) { return P; }

int main() {
  // expected-note@+1 {{'CapturedToDevice' defined here}}
  __float128 CapturedToDevice = 1;
  host_ok();
  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<variables, (lambda}}
    h.single_task<class variables>([=]() {
      // expected-error@+1 {{'__float128' is not supported on this target}}
      decltype(CapturedToDevice) D;
      // expected-error@+2 {{'CapturedToDevice' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
      // expected-error@+1 {{'__float128' is not supported on this target}}
      auto C = CapturedToDevice;
      // expected-note@+1 3{{used here}}
      Z<__float128> S;
      // expected-error@+1 {{'field1' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
      S.field1 += 1;
      // expected-error@+1 {{'field' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
      S.field = 1;
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall 4{{called by 'kernel_single_task<functions, (lambda}}
    h.single_task<class functions>([=]() {
      // expected-note@+1 2{{called by 'operator()'}}
      usage();
      // expected-note@+2 {{'BBBB' defined here}}
      // expected-error@+1 {{'__float128' is not supported on this target}}
      BIGTY BBBB;
      // expected-error@+4 {{'__float128' is not supported on this target}}
      // expected-note@+3 {{called by 'operator()'}}
      // expected-error@+2 2{{'foo' requires 128 bit size '__float128' type support, but device 'spir64' does not support it}}
      // expected-error@+1 {{'BBBB' requires 128 bit size 'BIGTY' (aka '__float128') type support, but device 'spir64' does not support it}}
      auto A = foo(BBBB);
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<ok, (lambda}}
    h.single_task<class ok>([=]() {
      // expected-note@+1 3{{used here}}
      Z<__float128> S;
      foo2<__float128>();
      auto A = sizeof(CapturedToDevice);
    });
  });

  return 0;
}
