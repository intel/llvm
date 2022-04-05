// Test checks Propagate Aspect Usage pass.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -debug-info-kind=constructor -dwarf-version=5 -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS-DBG -DPATH=%s

// CHECK-WARNINGS: warning: function 'checkStructUsesAspect(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-NEXT: note: the actual use is in funcWithStruct(int), compile with '-g' to get source location
// CHECK-WARNINGS-NEXT: note: which is called by checkStructUsesAspect(int), compile with '-g' to get source location
//
// CHECK-WARNINGS-DBG: warning: function 'checkStructUsesAspect(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-DBG-NEXT: note: the actual use is in funcWithStruct(int) at [[PATH]]:27:5
// CHECK-WARNINGS-DBG-NEXT: note: which is called by checkStructUsesAspect(int) at [[PATH]]:32:10

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

class KernelName;

struct [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] Struct {
  int a = 0;
};

int funcWithStruct(int) {
  Struct s;
  s.a = 1;
  return s.a;
}

[[sycl::device_has(aspect::fp64)]] int checkStructUsesAspect(int) {
  return funcWithStruct(1);
}

// Check that empty device_has() emits a warning.
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-EMPTY
// CHECK-EMPTY: warning: function 'checkEmptyDeviceHas()' uses aspect 'fp16' not listed in 'sycl::device_has()'
[[sycl::device_has()]] int checkEmptyDeviceHas() {
  return funcWithStruct(1);
}

[[sycl::device_has(aspect::fp16)]] int func2() {
  return funcWithStruct(1);
}

// Check that empty device_has() emits a warning despite the fact
// that invoked function's device_has() attribute is conformant
// with actual usage.
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-EMPTY2
// CHECK-EMPTY2: warning: function 'checkEmptyDeviceHas()' uses aspect 'fp16' not listed in 'sycl::device_has()'
[[sycl::device_has()]] int checkEmptyDeviceHas2() {
  return func2();
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

struct [[__sycl_detail__::__uses_aspects__(aspect::cpu)]] StructWithCpu {
  int a = 0;
};

int funcWithSeveralAspects() {
  Struct s1;
  StructWithCpu s2;
  return static_cast<double>(s1.a);
}

// Check that a warning diagnostic works in a case
// when there are several aspects present which conflict
// with declared in device_has().

// Check for fp64 aspect
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-SEVERAL-FP64
// CHECK-SEVERAL-FP64: warning: function 'checkSeveralAspects()' uses aspect 'fp64' not listed in 'sycl::device_has()'
// CHECK-SEVERAL-FP64-NEXT: note: the actual use is in funcWithSeveralAspects(), compile with '-g' to get source location
// CHECK-SEVERAL-FP64-NEXT: note: which is called by checkSeveralAspects(), compile with '-g' to get source location
//
// Check for cpu aspect
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-SEVERAL-CPU
// CHECK-SEVERAL-CPU: warning: function 'checkSeveralAspects()' uses aspect 'cpu' not listed in 'sycl::device_has()'
// CHECK-SEVERAL-CPU-NEXT: note: the actual use is in funcWithSeveralAspects(), compile with '-g' to get source location
// CHECK-SEVERAL-CPU-NEXT: note: which is called by checkSeveralAspects(), compile with '-g' to get source location
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
