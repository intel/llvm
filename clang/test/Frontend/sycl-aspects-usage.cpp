// Test checks Propagate Aspect Usage pass.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -debug-info-kind=constructor -dwarf-version=5 -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS-DBG -DPATH=%s

// CHECK-WARNINGS: warning: function 'checkStructUsesAspect(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-NEXT: note: the actual use is in checkStructUsesAspect(int), compile with '-g' to get source location
//
// CHECK-WARNINGS-DBG: warning: function 'checkStructUsesAspect(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-DBG-NEXT: note: the actual use is in checkStructUsesAspect(int) at [[PATH]]:25:5

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

class KernelName;

struct [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] Struct {
  int a = 0;
};

[[sycl::device_has(aspect::fp64)]] int checkStructUsesAspect(int) {
  Struct s;
  s.a = 1;
  return s.a;
}

[[sycl::device_has()]] int checkEmptyDeviceHas() {
  return 0;
}

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-DOUBLE
// CHECK-DOUBLE: warning: function 'checkDouble()' uses aspect 'fp64' not listed in 'sycl::device_has()'
// CHECK-DOUBLE-NEXT: note: the actual use is in checkDouble(), compile with '-g' to get source location

[[sycl::device_has(aspect::fp16)]] int checkDouble() {
  double d = 123;
  // Strange calculations just to prevent AST optimizations
  for (int i = 0; i < 10; ++i)
    d += d * d;

  return d;
}

double funcWithSeveralAspects() {
  Struct s;
  return static_cast<double>(s.a);
}

// Check that a warning diagnostic works in a case
// when there are several aspects present and a part
// of them conflicts with declared in device_has().
[[sycl::device_has(aspect::fp16)]] int checkSeveralAspects() {
  return funcWithSeveralAspects();
}

int main() {
  queue Q;
  Q.submit([&](handler &h) {
    h.single_task<KernelName>([=]() {
      checkStructUsesAspect(1);
      checkEmptyDeviceHas();
      checkDouble();
      checkSeveralAspects();
    });
  });
}
