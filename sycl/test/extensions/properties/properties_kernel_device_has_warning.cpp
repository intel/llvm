// RUN: %clangxx -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Tests for warnings when propagated aspects do not match the aspects available
// in a function, as specified through the 'sycl::device_has' property.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}Struct1::Struct1({{.*}})'}}
struct [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] Struct1 {
  int a = 0;
};

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func5(int)'}}
[[__sycl_detail__::__uses_aspects__(aspect::cpu)]] int func5(int a) {
  return 0;
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func3(int, int, int)'}}
int func3(int a, int b, int c) {
  Struct1 s;
  s.a = 1;
  return s.a;
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func4(int, int)'}}
int func4(int a, int b) {
  double x = 3.0;
  return func3(a, b, (int)x);
}

// expected-note-re@+1 2 {{propagated from call to function '{{.*}}func2(int, int)'}}
int func2(int a, int b) { return func3(a, b, 1); }

// expected-warning-re@+1 {{function '{{.*}}func1(int)' uses aspect 'fp16' not listed in its 'device_has' property}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((device_has<aspect::fp64>)) int func1(int a) {
  return func2(a, 1);
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func6(int, int, int)'}}
int func6(int a, int b, int c) { return func5(a); }

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}func7(int, int)'}}
int func7(int a, int b) {
  double x = 3.0;
  return func6(a, b, (int)x);
}

// expected-note-re@+1 2 {{propagated from call to function 'func8(int, int)'}}
int func8(int a, int b) { return func6(a, b, 1); }

// expected-warning-re@+1 {{function 'func9(int)' uses aspect 'cpu' not listed in its 'device_has' property}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((device_has<aspect::fp64>)) int func9(int a) {
  return func8(a, 1);
}

int main() {
  queue Q;
  Q.submit([&](handler &CGH) { CGH.single_task([=]() { int a = func1(1); }); });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = func2(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp16>},
                    [=]() { int a = func2(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp16, aspect::fp64>},
                    [=]() { int a = func2(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::fp16>},
                    [=]() { int a = func2(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+3 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::gpu>},
                    [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp16>},
                    [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp16, aspect::fp64>},
                    [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::fp16>},
                    [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::gpu, aspect::fp16, aspect::fp64>},
        [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::fp16, aspect::fp64, aspect::gpu>},
        [=]() { int a = func4(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{}, [=]() { int a = func9(1); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = func8(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::cpu>},
                    [=]() { int a = func8(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::cpu, aspect::fp64>},
                    [=]() { int a = func8(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::cpu>},
                    [=]() { int a = func8(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+3 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::gpu>},
                    [=]() { int a = func7(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::cpu>},
                    [=]() { int a = func7(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = func7(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::cpu, aspect::fp64>},
                    [=]() { int a = func7(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::cpu>},
                    [=]() { int a = func7(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::gpu, aspect::cpu, aspect::fp64>},
        [=]() { int a = func7(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::cpu, aspect::fp64, aspect::gpu>},
        [=]() { int a = func7(1, 2); });
  });
}
