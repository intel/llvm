// RUN: %clangxx -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// Tests for warnings when propagated aspects do not match the aspects available
// in a function, as specified through the 'sycl::device_has' property.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}StructWithFP16::StructWithFP16({{.*}})'}}
struct [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] StructWithFP16 {
  int a = 0;
};

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}funcWithCPU(int)'}}
[[__sycl_detail__::__uses_aspects__(aspect::cpu)]] int funcWithCPU(int a) {
  return 0;
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}funcUsingFP16(int, int, int)'}}
int funcUsingFP16(int a, int b, int c) {
  StructWithFP16 s;
  s.a = 1;
  return s.a;
}

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}funcUsingFP16AndFP64(int, int)'}}
int funcUsingFP16AndFP64(int a, int b) {
  double x = 3.0;
  return funcUsingFP16(a, b, (int)x);
}

// expected-note-re@+1 2 {{propagated from call to function '{{.*}}funcIndirectlyUsingFP16(int, int)'}}
int funcIndirectlyUsingFP16(int a, int b) { return funcUsingFP16(a, b, 1); }

// expected-warning-re@+2 {{function '{{.*}}funcUsingFP16HasFP64(int)' uses aspect 'fp16' not listed in its 'device_has' property}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((device_has<aspect::fp64>))
int funcUsingFP16HasFP64(int a) { return funcIndirectlyUsingFP16(a, 1); }

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}funcUsingCPU(int, int, int)'}}
int funcUsingCPU(int a, int b, int c) { return funcWithCPU(a); }

// expected-note-re@+1 4 {{propagated from call to function '{{.*}}funcUsingCPUAndFP64(int, int)'}}
int funcUsingCPUAndFP64(int a, int b) {
  double x = 3.0;
  return funcUsingCPU(a, b, (int)x);
}

// expected-note-re@+1 2 {{propagated from call to function '{{.*}}funcIndirectlyUsingCPU(int, int)'}}
int funcIndirectlyUsingCPU(int a, int b) { return funcUsingCPU(a, b, 1); }

// expected-warning-re@+2 {{function '{{.*}}funcUsingCPUHasFP64(int)' uses aspect 'cpu' not listed in its 'device_has' property}}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((device_has<aspect::fp64>))
int funcUsingCPUHasFP64(int a) { return funcIndirectlyUsingCPU(a, 1); }

int main() {
  queue Q;
  Q.submit([&](handler &CGH) {
    CGH.single_task([=]() { int a = funcUsingFP16HasFP64(1); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = funcIndirectlyUsingFP16(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp16>},
                    [=]() { int a = funcIndirectlyUsingFP16(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp16, aspect::fp64>},
                    [=]() { int a = funcIndirectlyUsingFP16(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::fp16>},
                    [=]() { int a = funcIndirectlyUsingFP16(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+3 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::gpu>},
                    [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp16>},
                    [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::gpu>}, [=]() {
      if constexpr (false) {
        int a = funcUsingFP16AndFP64(1, 2);
      }
    });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp16, aspect::fp64>},
                    [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::fp16>},
                    [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::gpu, aspect::fp16, aspect::fp64>},
        [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::fp16, aspect::fp64, aspect::gpu>},
        [=]() { int a = funcUsingFP16AndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{}, [=]() { int a = funcUsingCPUHasFP64(1); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = funcIndirectlyUsingCPU(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::cpu>},
                    [=]() { int a = funcIndirectlyUsingCPU(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::cpu, aspect::fp64>},
                    [=]() { int a = funcIndirectlyUsingCPU(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::cpu>},
                    [=]() { int a = funcIndirectlyUsingCPU(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+3 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::gpu>},
                    [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::cpu>},
                    [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
    CGH.single_task(properties{device_has<aspect::fp64>},
                    [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::gpu>}, [=]() {
      if constexpr (false) {
        int a = funcUsingCPUAndFP64(1, 2);
      }
    });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::cpu, aspect::fp64>},
                    [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(properties{device_has<aspect::fp64, aspect::cpu>},
                    [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::gpu, aspect::cpu, aspect::fp64>},
        [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        properties{device_has<aspect::cpu, aspect::fp64, aspect::gpu>},
        [=]() { int a = funcUsingCPUAndFP64(1, 2); });
  });
}
