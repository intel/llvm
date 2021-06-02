// RUN: %clang_cc1 -triple spir64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:    -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s

#include "sycl.hpp"

sycl::queue deviceQueue;

typedef __uint128_t BIGTY;

template <class T>
class Z {
public:
  // expected-error@+2 3{{'__int128' is not supported on this target}}
  // expected-note@+1 {{'field' defined here}}
  T field;
  // expected-error@+2 3{{'__int128' is not supported on this target}}
  // expected-note@+1 2{{'field1' defined here}}
  __int128 field1;
  using BIGTYPE = __int128;
  // expected-error@+2 3{{'__int128' is not supported on this target}}
  // expected-note@+1 {{'bigfield' defined here}}
  BIGTYPE bigfield;
};

void host_ok(void) {
  __int128 A;
  int B = sizeof(__int128);
  Z<__int128> C;
  C.field1 = A;
}

void usage() {
  // expected-error@+2 {{'__int128' is not supported on this target}}
  // expected-note@+1 3{{'A' defined here}}
  __int128 A;
  // expected-note@+1 3{{used here}}
  Z<__int128> C;
  // expected-error@+2 {{'A' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
  // expected-error@+1 {{'field1' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
  C.field1 = A;
  // expected-error@+1 {{'bigfield' requires 128 bit size 'Z::BIGTYPE' (aka '__int128') type support, but device 'spir64' does not support it}}
  C.bigfield += 1.0;

  // expected-note@+2 1{{used here}}
  // expected-error@+1 {{'A' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
  auto foo1 = [=]() {
    // expected-error@+1 {{'__int128' is not supported on this target}}
    __int128 AA;
    // expected-error@+3 2{{'__int128' is not supported on this target}}
    // expected-note@+2 {{'BB' defined here}}
    // expected-error@+1 {{'A' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
    auto BB = A;
    // expected-error@+1 {{'BB' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
    BB += 1;
  };

  // expected-note@+1 {{called by 'usage'}}
  foo1();
}

template <typename t>
void foo2(){};

// expected-note@+3 {{'P' defined here}}
// expected-error@+2 {{'P' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
// expected-note@+1 2{{'foo' defined here}}
__int128 foo(__int128 P) { return P; }

int foobar() {
  // expected-note@+1 {{'operator __int128' defined here}}
  struct X { operator  __int128() const{return 0;}; } x;
  bool a = false;
  // expected-error@+1 {{'operator __int128' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
  a = x == __int128(0);
  return a;
}

int main() {
  // expected-note@+1 {{'CapturedToDevice' defined here}}
  __int128 CapturedToDevice = 1;
  host_ok();
  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<variables, (lambda}}
    h.single_task<class variables>([=]() {
      // expected-error@+1 {{'__int128' is not supported on this target}}
      decltype(CapturedToDevice) D;
      // expected-error@+2 {{'__int128' is not supported on this target}}
      // expected-error@+1 {{'CapturedToDevice' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
      auto C = CapturedToDevice;
      // expected-note@+1 3{{used here}}
      Z<__int128> S;
      // expected-error@+1 {{'field1' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
      S.field1 += 1;
      // expected-error@+1 {{'field' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
      S.field = 1;
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall 5{{called by 'kernel_single_task<functions, (lambda}}
    h.single_task<class functions>([=]() {
      // expected-note@+1 2{{called by 'operator()'}}
      usage();
      // expected-error@+2 {{'unsigned __int128' is not supported on this target}}
      // expected-note@+1 {{'BBBB' defined here}}
      BIGTY BBBB;
      // expected-error@+4 {{'__int128' is not supported on this target}}
      // expected-error@+3 {{'BBBB' requires 128 bit size 'BIGTY' (aka 'unsigned __int128') type support, but device 'spir64' does not support it}}
      // expected-error@+2 2{{'foo' requires 128 bit size '__int128' type support, but device 'spir64' does not support it}}
      // expected-note@+1 1{{called by 'operator()'}}
      auto A = foo(BBBB);
      // expected-note@+1 {{called by 'operator()'}}
      auto i = foobar();
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<ok, (lambda}}
    h.single_task<class ok>([=]() {
      // expected-note@+1 3{{used here}}
      Z<__int128> S;
      foo2<__int128>();
      auto A = sizeof(CapturedToDevice);
    });
  });

  return 0;
}

// no error expected
BIGTY zoo(BIGTY h) {
  h = 1;
  return h;
}

namespace PR12964 {
  struct X { operator  __int128() const; } x;
  bool a = x == __int128(0);
}

