// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -Wno-sycl-work-group-dtor -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=GLOBAL < %t.ll
// RUN: FileCheck %s --implicit-check-not=__cxa_atexit --implicit-check-not=_ZGV --implicit-check-not=llvm.global_ctors < %t.ll

// This test checks the code generated for variables with a non-trivial
// destructor declared at work group scope, which is diagnosed by
// -Wsycl-work-group-dtor. These variables are placed in local memory, but keep
// automatic storage duration: their destructor is called at the end of their
// scope, and is never registered with __cxa_atexit. Variables at work group scope that do not need a destructor are checked in
// pfwg-locals.cpp.
//
// The SYCL lowering pass (LowerWGScope) always runs in clang for SYCL device
// compilation, so the checks are on the lowered code: work group scope code is
// executed by the work group leader only, between barriers.
//
// The globals are checked with a separate prefix so the GLOBAL-DAG lines can
// be placed next to the declarations they match.

#include "sycl.hpp"

struct NonTrivial {
  int I;
  ~NonTrivial() {}
};

void wg_func(sycl::group<1> G) {
  // GLOBAL-DAG: @_ZZ7wg_funcN4sycl3_V15groupILi1EEEE6InFunc = internal addrspace(3) global %struct.NonTrivial undef, align 4
  NonTrivial InFunc;
  G.parallel_for_work_item();
}

// CHECK-LABEL: define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_5groupILi1EEEE_clES5_(
// CHECK-SAME: !work_group_scope
// The destructors are called by the work group leader, between barriers.
// The variable in the loop body is destroyed at the end of every iteration.
// CHECK: for.body:
// CHECK: call spir_func void @_Z22__spirv_ControlBarrieriii(
// CHECK-NEXT: %[[CMP:cmpz[0-9]+]] = icmp eq i64
// CHECK-NEXT: br i1 %[[CMP]], label %[[LEADER:wg_leader[0-9]+]], label %[[CF:wg_cf[0-9]+]]
// CHECK: [[LEADER]]:
// CHECK-NEXT: call spir_func void @_ZN10NonTrivialD1Ev(ptr addrspace(4) {{.*}}addrspacecast (ptr addrspace(3) @_ZZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_5groupILi1EEEE_clES5_E6InLoop to ptr addrspace(4)))
// CHECK-NEXT: br label %[[CF]]
// CHECK: [[CF]]:
// CHECK-NEXT: call spir_func void @_Z22__spirv_ControlBarrieriii(
// CHECK: for.end:
// CHECK: call spir_func void @_Z22__spirv_ControlBarrieriii(
// CHECK-NEXT: %[[CMP:cmpz[0-9]+]] = icmp eq i64
// CHECK-NEXT: br i1 %[[CMP]], label %[[LEADER:wg_leader[0-9]+]], label %[[CF:wg_cf[0-9]+]]
// CHECK: [[LEADER]]:
// CHECK: call spir_func void @_ZN10NonTrivialD1Ev(ptr addrspace(4) {{.*}}addrspacecast (ptr addrspace(3) @_ZZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_5groupILi1EEEE_clES5_E2NT to ptr addrspace(4)))
// CHECK-NEXT: br label %[[CF]]
// CHECK: [[CF]]:
// CHECK-NEXT: call spir_func void @_Z22__spirv_ControlBarrieriii(
// CHECK-NEXT: ret void

int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class Kernel>(
        sycl::range<1>{1}, sycl::range<1>{1}, [=](sycl::group<1> G) {
          // GLOBAL-DAG: @{{.*}}E2NT = internal addrspace(3) global %struct.NonTrivial undef, align 4
          NonTrivial NT;
          for (int J = 0; J < 2; ++J) {
            // GLOBAL-DAG: @{{.*}}E6InLoop = internal addrspace(3) global %struct.NonTrivial undef, align 4
            NonTrivial InLoop;
          }
          // FIXME: Captures with a non-trivial destructor are not tested: the
          // kernel function is not lowered by LowerWGScope, so the destructor
          // of the kernel object is called by every work item.
          // See https://github.com/intel/llvm/issues/23700
        });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class KernelWGFunc>(
        sycl::range<1>{1}, sycl::range<1>{1},
        [=](sycl::group<1> G) { wg_func(G); });
  });
  return 0;
}

// A work group function other than the parallel_for_work_group lambda. It has
// work group scope because it calls parallel_for_work_item. It is called from
// a separate kernel because LowerWGScope does not lower a work group function
// that calls a work group function but does not call parallel_for_work_item.
// CHECK-LABEL: define linkonce_odr spir_func void @_Z7wg_funcN4sycl3_V15groupILi1EEE(
// CHECK-SAME: !work_group_scope
// The variable is destroyed by the work group leader after the work items are
// done with parallel_for_work_item.
// CHECK: call spir_func void @_ZN4sycl3_V15groupILi1EE22parallel_for_work_itemEv(
// CHECK: call spir_func void @_Z22__spirv_ControlBarrieriii(
// CHECK-NEXT: %cmpz = icmp eq i64
// CHECK-NEXT: br i1 %cmpz, label %wg_leader, label %wg_cf
// CHECK: wg_leader:
// CHECK-NEXT: call spir_func void @_ZN10NonTrivialD1Ev(ptr addrspace(4) {{.*}}addrspacecast (ptr addrspace(3) @_ZZ7wg_funcN4sycl3_V15groupILi1EEEE6InFunc to ptr addrspace(4)))
// CHECK-NEXT: br label %wg_cf
// CHECK: wg_cf:
// CHECK-NEXT: call spir_func void @_Z22__spirv_ControlBarrieriii(
// CHECK-NEXT: ret void
