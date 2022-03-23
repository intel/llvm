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

// CHECK-WARNINGS: warning: function 'func1(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-NEXT: note: the actual use is in func3(int, int, int), compile with '-g' to get source location
// CHECK-WARNINGS-NEXT: note: which is called by func2(int, int), compile with '-g' to get source location
// CHECK-WARNINGS-NEXT: note: which is called by func1(int), compile with '-g' to get source location
//
// CHECK-WARNINGS-DBG: warning: function 'func1(int)' uses aspect 'fp16' not listed in 'sycl::device_has()'
// CHECK-WARNINGS-DBG-NEXT: note: the actual use is in func3(int, int, int) at [[PATH]]:45:5
// CHECK-WARNINGS-DBG-NEXT: note: which is called by func2(int, int) at [[PATH]]:49:34
// CHECK-WARNINGS-DBG-NEXT: note: which is called by func1(int) at [[PATH]]:51:62

// Check that PropagateAspectUsage pass do not wipes out sycl_explicit_simd
// metadata.
// RUN: %clangxx -fsycl %s -emit-llvm -S -o - | FileCheck %s --check-prefix CHECK-ESIMD

// CHECK-ESIMD: @_Z5func4v() {{.*}} !sycl_explicit_simd {{.*}} !intel_used_aspects
// CHECK-ESIMD: @_ZTSZ4mainEUlT_E0_() {{.*}} !sycl_explicit_simd {{.*}} !intel_used_aspects

#include <CL/sycl.hpp>

#include <sycl/ext/intel/esimd.hpp>

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

SYCL_ESIMD_KERNEL int func4() {
  Struct s;
  s.a = 1;
  return s.a;
}

int main() {
  queue Q;
  Q.parallel_for(1, [=](auto i) { int a = func1(1); });
  Q.parallel_for(1, [=](auto i) SYCL_ESIMD_KERNEL { int a = func4(); });
}
