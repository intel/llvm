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

constexpr auto props_64 = properties{device_has<aspect::fp64>};
constexpr auto props_16 = properties{device_has<aspect::fp16>};
constexpr auto props_1664 = properties{device_has<aspect::fp16, aspect::fp64>};
constexpr auto props_6416 = properties{device_has<aspect::fp64, aspect::fp16>};
constexpr auto props_gpu = properties{device_has<aspect::gpu>};
constexpr auto props_gpu1664 =
    properties{device_has<aspect::gpu, aspect::fp16, aspect::fp64>};
constexpr auto props_1664gpu =
    properties{device_has<aspect::fp16, aspect::fp64, aspect::gpu>};
constexpr auto props_emp = properties{};
constexpr auto props_cpu = properties{device_has<aspect::cpu>};
constexpr auto props_cpu64 = properties{device_has<aspect::cpu, aspect::fp64>};
constexpr auto props_64cpu = properties{device_has<aspect::fp64, aspect::cpu>};
constexpr auto props_gpucpu64 =
    properties{device_has<aspect::gpu, aspect::cpu, aspect::fp64>};
constexpr auto props_cpu64gpu =
    properties{device_has<aspect::cpu, aspect::fp64, aspect::gpu>};

template <typename T> struct K_funcIndirectlyUsingFP16 {
  T *Props;
  K_funcIndirectlyUsingFP16(T Props_param) { Props = &Props_param; };
  void operator()() const { int a = funcIndirectlyUsingFP16(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcIndirectlyUsingFP16_Warn16 {
  T *Props;
  K_funcIndirectlyUsingFP16_Warn16(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
  void operator()() const { int a = funcIndirectlyUsingFP16(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingFP16AndFP64 {
  T *Props;
  K_funcUsingFP16AndFP64(T Props_param) { Props = &Props_param; };
  void operator()() const { int a = funcUsingFP16AndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingFP16AndFP64_Warn16 {
  T *Props;
  K_funcUsingFP16AndFP64_Warn16(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
  void operator()() const { int a = funcUsingFP16AndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingFP16AndFP64_Warn64 {
  T *Props;
  K_funcUsingFP16AndFP64_Warn64(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
  void operator()() const { int a = funcUsingFP16AndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingFP16AndFP64_Warn1664 {
  T *Props;
  K_funcUsingFP16AndFP64_Warn1664(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'fp16' not listed in its 'device_has' property}}
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
  void operator()() const { int a = funcUsingFP16AndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingFP16AndFP64_False {
  T *Props;
  K_funcUsingFP16AndFP64_False(T Props_param) { Props = &Props_param; };
  void operator()() const {
    if constexpr (false) {
      int a = funcUsingFP16AndFP64(1, 2);
    }
  }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingCPUHasFP64 {
  T *Props;
  K_funcUsingCPUHasFP64(T Props_param) { Props = &Props_param; };
  void operator()() const { int a = funcUsingCPUHasFP64(1); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcIndirectlyUsingCPU {
  T *Props;
  K_funcIndirectlyUsingCPU(T Props_param) { Props = &Props_param; };
  void operator()() const { int a = funcIndirectlyUsingCPU(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcIndirectlyUsingCPU_WarnCPU {
  T *Props;
  K_funcIndirectlyUsingCPU_WarnCPU(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
  void operator()() const { int a = funcIndirectlyUsingCPU(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingCPUAndFP64 {
  T *Props;
  K_funcUsingCPUAndFP64(T Props_param) { Props = &Props_param; };
  void operator()() const { int a = funcUsingCPUAndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingCPUAndFP64_WarnCPU {
  T *Props;
  K_funcUsingCPUAndFP64_WarnCPU(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
  void operator()() const { int a = funcUsingCPUAndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingCPUAndFP64_Warn64 {
  T *Props;
  K_funcUsingCPUAndFP64_Warn64(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
  void operator()() const { int a = funcUsingCPUAndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingCPUAndFP64_Warn64CPU {
  T *Props;
  K_funcUsingCPUAndFP64_Warn64CPU(T Props_param) { Props = &Props_param; };
  // expected-warning-re@+2 {{function '{{.*}}' uses aspect 'cpu' not listed in its 'device_has' property}}
  // expected-warning-re@+1 {{function '{{.*}}' uses aspect 'fp64' not listed in its 'device_has' property}}
  void operator()() const { int a = funcUsingCPUAndFP64(1, 2); }
  auto get(properties_tag) { return *Props; }
};

template <typename T> struct K_funcUsingCPUAndFP64_False {
  T *Props;
  K_funcUsingCPUAndFP64_False(T Props_param) { Props = &Props_param; };
  void operator()() const {
    if constexpr (false) {
      int a = funcUsingCPUAndFP64(1, 2);
    }
  }
  auto get(properties_tag) { return *Props; }
};

int main() {
  queue Q;
  Q.submit([&](handler &CGH) {
    CGH.single_task([=]() { int a = funcUsingFP16HasFP64(1); });
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcIndirectlyUsingFP16_Warn16<decltype(props_64)>(props_64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(K_funcIndirectlyUsingFP16<decltype(props_16)>(props_16));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcIndirectlyUsingFP16<decltype(props_1664)>(props_1664));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcIndirectlyUsingFP16<decltype(props_6416)>(props_6416));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn1664<decltype(props_gpu)>(props_gpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn64<decltype(props_16)>(props_16));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn16<decltype(props_64)>(props_64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_False<decltype(props_gpu)>(props_gpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn16<decltype(props_1664)>(props_1664));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn16<decltype(props_6416)>(props_6416));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn16<decltype(props_gpu1664)>(props_gpu1664));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingFP16AndFP64_Warn16<decltype(props_1664gpu)>(props_1664gpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(K_funcUsingCPUHasFP64<decltype(props_emp)>(props_emp));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcIndirectlyUsingCPU_WarnCPU<decltype(props_64)>(props_64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(K_funcIndirectlyUsingCPU<decltype(props_cpu)>(props_cpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcIndirectlyUsingCPU<decltype(props_cpu64)>(props_cpu64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcIndirectlyUsingCPU<decltype(props_64cpu)>(props_64cpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingCPUAndFP64_Warn64CPU<decltype(props_gpu)>(props_gpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingCPUAndFP64_Warn64<decltype(props_cpu)>(props_cpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingCPUAndFP64_WarnCPU<decltype(props_64)>(props_64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingCPUAndFP64_False<decltype(props_gpu)>(props_gpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(K_funcUsingCPUAndFP64<decltype(props_cpu64)>(props_cpu64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(K_funcUsingCPUAndFP64<decltype(props_64cpu)>(props_64cpu));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingCPUAndFP64<decltype(props_gpucpu64)>(props_gpucpu64));
  });
  Q.submit([&](handler &CGH) {
    CGH.single_task(
        K_funcUsingCPUAndFP64<decltype(props_cpu64gpu)>(props_cpu64gpu));
  });
}
