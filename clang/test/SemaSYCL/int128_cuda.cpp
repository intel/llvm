// RUN: %clang_cc1 -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:    -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s

#include "sycl.hpp"

// expected-no-diagnostics

sycl::queue deviceQueue;

typedef __uint128_t BIGTY;

template <class T>
class Z {
public:
  T field;
  __int128 field1;
  using BIGTYPE = __int128;
  BIGTYPE bigfield;
};

void host_ok(void) {
  __int128 A;
  int B = sizeof(__int128);
  Z<__int128> C;
  C.field1 = A;
}

void usage() {
  __int128 A;
  Z<__int128> C;
  C.field1 = A;
  C.bigfield += 1.0;

  auto foo1 = [=]() {
    __int128 AA;
    auto BB = A;
    BB += 1;
  };

  foo1();
}

template <typename t>
void foo2(){};

__int128 foo(__int128 P) { return P; }

int foobar() {
  struct X { operator  __int128() const{return 0;}; } x;
  bool a = false;
  a = x == __int128(0);
  return a;
}

int main() {
  __int128 CapturedToDevice = 1;
  host_ok();
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class variables>([=]() {
      decltype(CapturedToDevice) D;
      auto C = CapturedToDevice;
      Z<__int128> S;
      S.field1 += 1;
      S.field = 1;
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class functions>([=]() {
      usage();
      BIGTY BBBB;
      auto A = foo(BBBB);
      auto i = foobar();
    });
  });

  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class ok>([=]() {
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

