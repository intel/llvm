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

// CHECK-WARNINGS: warning: function '_Z5func1v' uses aspect 'fp16' not listed in `sycl::device_has()`
// CHECK-WARNINGS-NEXT: use is from this call chain:
// CHECK-WARNINGS-NEXT:  func1()
// CHECK-WARNINGS-NEXT:  func2()
// CHECK-WARNINGS-NEXT:  func3()
// CHECK-WARNINGS-NEXT: compile with '-g' to get source location

// CHECK-WARNINGS-DBG: warning: function '_Z5func1v' uses aspect 'fp16' not listed in `sycl::device_has()`
// CHECK-WARNINGS-DBG-NEXT: use is from this call chain:
// CHECK-WARNINGS-DBG-NEXT:  func1() (defined at [[PATH]]:45:57)
// CHECK-WARNINGS-DBG-NEXT:  func2() (defined at [[PATH]]:43:22)
// CHECK-WARNINGS-DBG-NEXT:  func3() (defined at [[PATH]]:39:5)

#include <CL/sycl.hpp>

using namespace cl::sycl;

struct [[__sycl_detail__::__uses_aspects__(aspect::fp16)]] Struct {
  int a = 0;
};

int func3() {
  Struct s;
  s.a = 1;
  return s.a;
}

int func2() { return func3(); }

[[sycl::device_has(aspect::fp64)]] int func1() { return func2(); }

int main() {
  queue Q;
  Q.parallel_for(1, [=](auto i) { int a = func1(); });
}
