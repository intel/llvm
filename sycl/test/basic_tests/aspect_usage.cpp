// Test checks Propagate Aspect Usage pass.

// RUN: %clangxx -fsycl -fno-legacy-pass-manager %s -emit-llvm -S -o - | FileCheck %s --check-prefix CHECK-NODES
// RUN: %clangxx -fsycl -fno-legacy-pass-manager -fno-sycl-early-optimizations %s -emit-llvm -S -o - | FileCheck %s --check-prefix CHECK-NODES
// RUN: %clangxx -fsycl -flegacy-pass-manager %s -emit-llvm -S -o - | FileCheck %s --check-prefix CHECK-NODES
// RUN: %clangxx -fsycl -flegacy-pass-manager -fno-sycl-early-optimizations %s -emit-llvm -S -o - | FileCheck %s --check-prefix CHECK-NODES

// fp16 aspect corresponds to constant 5.
// CHECK-NODES-DAG: !intel_used_aspects ![[NODE:[0-9]+]]
// CHECK-NODES: ![[NODE]] = !{i32 5}

// RUN: %clangxx -fsycl %s -emit-llvm -S -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS

// RUN: %clangxx -fsycl %s -g -emit-llvm -S -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-WARNINGS-DBG -DPATH=%s

// CHECK-WARNINGS: warning: function 'func1(int)' uses aspect 'fp16' not listed in `sycl::device_has()`
// CHECK-WARNINGS-NEXT: use is from this call chain:
// CHECK-WARNINGS-NEXT:  func1(int)
// CHECK-WARNINGS-NEXT:  func2(int, int)
// CHECK-WARNINGS-NEXT:  func3(int, int, int)
// CHECK-WARNINGS-NEXT: compile with '-g' to get source location

// CHECK-WARNINGS-DBG: warning: function 'func1(int)' uses aspect 'fp16' not listed in `sycl::device_has()`
// CHECK-WARNINGS-DBG-NEXT: use is from this call chain:
// CHECK-WARNINGS-DBG-NEXT:  func1(int) (defined at [[PATH]]:45:62)
// CHECK-WARNINGS-DBG-NEXT:  func2(int, int) (defined at [[PATH]]:43:34)
// CHECK-WARNINGS-DBG-NEXT:  func3(int, int, int) (defined at [[PATH]]:39:5)

#include <CL/sycl.hpp>

using namespace cl::sycl;

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
  Q.parallel_for(1, [=](auto i) { int a = func1(1); });
}
