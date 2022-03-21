// Test checks Propagate Aspect Usage pass.

// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS

// RUNx: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown %s -debug-info-kind=constructor -dwarf-version=5 -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS-DBG -DPATH=%s

// CHECK-WARNINGS: warning: function 'func1(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-NEXT: note: the actual use is in func3(int, int, int), compile with '-g' to get source location
// CHECK-WARNINGS-NEXT: note: which is called by func2(int, int), compile with '-g' to get source location
// CHECK-WARNINGS-NEXT: note: which is called by func1(int), compile with '-g' to get source location
//
// CHECK-WARNINGS-DBG: warning: function 'func1(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-DBG-NEXT: note: the actual use is in func3(int, int, int) at [[PATH]]:36:5
// CHECK-WARNINGS-DBG-NEXT: note: which is called by func2(int, int) at [[PATH]]:40:34
// CHECK-WARNINGS-DBG-NEXT: note: which is called by func1(int) at [[PATH]]:42:62

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

class KernelName;

struct [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] Struct {
  int a = 0;
};

int func3(int a, int b, int c) {
  Struct s;
  s.a = 1;
  return s.a;
}

int func2(int a, int b) { return func3(a, b, 1); }

[[sycl::device_has(aspect::fp64)]] int func1(int a) { return func2(a, 1); }

int main() {
  queue Q;
  Q.submit([&](handler &h) {
    h.single_task<KernelName>([=]() { int a = func1(1); });
  });
}
