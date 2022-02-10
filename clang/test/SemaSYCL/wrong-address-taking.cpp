// RUN: %clang_cc1 -fsycl-is-device -fsycl-allow-func-ptr -internal-isystem %S/Inputs -fsyntax-only -verify -sycl-std=2020 -std=c++17 %s

#include "sycl.hpp"

int badFoo(int P) {
  return P + 2;
}

[[intel::device_indirectly_callable]] int goodFoo(int P) {
  return P + 2;
}

SYCL_EXTERNAL float externalBadFoo(int P);
[[intel::device_indirectly_callable]] unsigned externalGoodFoo(int P);

sycl::queue myQueue;

SYCL_EXTERNAL int runFn(int (&)(int));
SYCL_EXTERNAL int runFn1(int (*)(int));

struct ForMembers {
  [[intel::device_indirectly_callable]] int goodMember(int) { return 1; }
  int badMember(int) { return 2; }

  static int badStaticMember(int) { return 2; }
};

template <typename Fn, typename... Args> void templateCaller(Fn F, Args... As) {
  F(As...);
}

template <auto Fn, typename... Args> void templateCaller1(Args... As) {
  // expected-error@+1 2{{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  Fn(As...);
  // expected-error@+1 2{{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  runFn(*Fn);
}

void basicUsage() {
  // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  int (*p)(int) = &badFoo;
  // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  int (*p2)(int) = badFoo;
}

template <typename T> void templatedContext() {

  // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  int (*p)(int) = &badFoo;
  // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  auto p1 = &ForMembers::badMember;

  // expected-error@+2 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
  // expected-note@+1 {{called by 'templatedContext<int>'}}
  templateCaller1<badFoo>(1);
}

int main() {

  myQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall 2{{called by 'kernel_single_task<Basic}}
    h.single_task<class Basic>(
        [=]() {
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          int (*p)(int) = &badFoo;
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          int (*p2)(int) = badFoo;

          // OK
          int (*p3)(int) = &goodFoo;
          int (*p4)(int) = goodFoo;

          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          auto p5 = &externalBadFoo;
          auto *p6 = &externalGoodFoo;

          // Make sure that assignment is diagnosed correctly;
          int (*a)(int);
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          a = badFoo;
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          a = &badFoo;

          a = goodFoo;
          a = &goodFoo;

          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          constexpr auto b = badFoo;
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          constexpr auto c = &badFoo;
          // expected-note@+1 {{called by 'operator()'}}
          basicUsage();
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<Members}}
    h.single_task<class Members>(
        [=]() {
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          auto p = &ForMembers::badMember;
          auto p1 = &ForMembers::goodMember;
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          auto *p2 = &ForMembers::badStaticMember;
        });
  });

  myQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall 2{{called by 'kernel_single_task<RunVia}}
    h.single_task<class RunVia>(
        [=]() {
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          int baz = runFn(badFoo);

          baz = runFn(goodFoo);

          // expected-error@+1 2{{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          baz = runFn1(badFoo);

          baz = runFn1(goodFoo);

          // expected-error@+1 2{{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          templateCaller(badFoo, 2);
          templateCaller(goodFoo, 1);

          templateCaller1<goodFoo>(1);

          // expected-note@+2 {{called by 'operator()'}}
          // expected-error@+1 {{taking address of a function not marked with 'intel::device_indirectly_callable' attribute is not allowed in SYCL device code}}
          templateCaller1<badFoo>(1);
        });
  });
  myQueue.submit([&](sycl::handler &h) {
    // expected-note@#KernelSingleTaskKernelFuncCall {{called by 'kernel_single_task<RunTemplatedContext}}
    h.single_task<class RunTemplatedContext>(
        [=]() {
          // expected-note@+1 {{called by 'operator()'}}
          templatedContext<int>();
        });
  });
  return 0;
}
