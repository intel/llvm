// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=GLOBAL < %t.ll
// RUN: FileCheck %s --implicit-check-not=__cxa_atexit --implicit-check-not=_ZGV --implicit-check-not=llvm.global_ctors < %t.ll

// This test checks the code generated for variables declared at work group
// scope in a parallel_for_work_group kernel that do not need a work group
// scope destructor: trivially destructible variables and captures,
// work_group_static objects, private_memory objects and temporaries. Variables
// with a non-trivial destructor are checked in pfwg-locals-dtor.cpp.
//
// The SYCL lowering pass (LowerWGScope) always runs in clang for SYCL device
// compilation, so the checks are on the lowered code: work group scope code is
// executed by the work group leader only, between barriers.
//
// The globals are checked with a separate prefix so the GLOBAL-DAG lines can
// be placed next to the declarations they match. The --implicit-check-not
// options are on the CHECK run, which has no CHECK-DAG lines, because
// FileCheck does not apply them between the matches of a CHECK-DAG group.

#include "sycl.hpp"

template <typename T> class [[__sycl_detail__::wg_scope]] work_group_static {
public:
  work_group_static() = default;
  work_group_static(const work_group_static &) = delete;
  work_group_static &operator=(const work_group_static &) = delete;
  operator T &() noexcept { return data; }

private:
  T data;
};

namespace sycl {
template <typename T, int Dimensions = 1>
class [[__sycl_detail__::sycl_type(private_memory)]] private_memory {
public:
  private_memory(const group<Dimensions> &) {}

private:
  T Val;
};
} // namespace sycl

struct Pod {
  int I;
};

struct NonTrivial {
  int I;
  ~NonTrivial() {}
};

NonTrivial makeNonTrivial() { return NonTrivial{}; }

// The kernel object, which holds the captures, is placed in local memory. Its
// captures are trivially destructible, so no destructor is registered for it.
// GLOBAL-DAG: @{{.*}}Kernel3PodE12__SYCLKernel = internal addrspace(3) global %class.anon undef, align 4
// CHECK-LABEL: define weak_odr spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E6Kernel(
// CHECK-SAME: !work_group_scope
// CHECK: call void @llvm.memcpy.{{.*}}(ptr addrspace(4) align 4 addrspacecast (ptr addrspace(3) @{{.*}}Kernel3PodE12__SYCLKernel to ptr addrspace(4)),
// CHECK: call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_5groupILi1EEEE_clES5_(ptr addrspace(4) {{.*}}addrspacecast (ptr addrspace(3) @{{.*}}Kernel3PodE12__SYCLKernel to ptr addrspace(4)),
// CHECK: ret void

// CHECK-LABEL: define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_5groupILi1EEEE_clES5_(
// CHECK-SAME: !work_group_scope
// private_memory objects have work item scope: they are allocas, not globals.
// CHECK: %PM = alloca %"class.sycl::private_memory", align 4, !work_item_scope
// CHECK: %ref.tmp = alloca %struct.NonTrivial, align 4
// CHECK: %agg.tmp.ensured = alloca %struct.NonTrivial, align 4
// All of the work group scope code, including the construction and
// destruction of the private_memory object and of the temporaries, is
// executed by the work group leader.
// CHECK: wg_leader:
// CHECK: store i32 1, ptr addrspace(4) addrspacecast (ptr addrspace(3) @{{.*}}E6Scalar to ptr addrspace(4)), align 4
// CHECK: store i32 2, ptr addrspace(4) addrspacecast (ptr addrspace(3) @{{.*}}E1P to ptr addrspace(4)), align 4
// CHECK: call spir_func void @_ZN4sycl14private_memoryI10NonTrivialLi1EEC1ERKNS_3_V15groupILi1EEE(ptr addrspace(4) {{.*}}%PM.ascast,
// The lifetime extended temporary is an alloca of the leader. Only the
// reference bound to it is in local memory.
// CHECK: call spir_func void @_Z14makeNonTrivialv(ptr {{.*}}sret(%struct.NonTrivial) align 4 %ref.tmp.ascast.ascast)
// CHECK: store ptr addrspace(4) %ref.tmp.ascast, ptr addrspace(4) addrspacecast (ptr addrspace(3) @{{.*}}E3Ext to ptr addrspace(4)), align 8
// A discarded temporary is destroyed at the end of the full-expression.
// CHECK: call spir_func void @_Z14makeNonTrivialv(ptr {{.*}}sret(%struct.NonTrivial) align 4 %agg.tmp.ensured.ascast.ascast)
// CHECK-NEXT: call spir_func void @_ZN10NonTrivialD1Ev(ptr addrspace(4) {{.*}}%agg.tmp.ensured.ascast)
// CHECK: call {{.*}}@_ZN17work_group_staticIiEcvRiEv({{.*}}addrspacecast (ptr addrspace(3) @{{.*}}E3WGS to ptr addrspace(4)))
// CHECK: call {{.*}}@_ZN17work_group_staticIiEcvRiEv({{.*}}addrspacecast (ptr addrspace(3) @{{.*}}E9StaticWGS to ptr addrspace(4)))
// The lifetime extended temporary and the private_memory object are destroyed
// at the end of the scope, in reverse order of construction.
// CHECK: call spir_func void @_ZN10NonTrivialD1Ev(ptr addrspace(4) {{.*}}%ref.tmp.ascast)
// CHECK-NEXT: call spir_func void @_ZN4sycl14private_memoryI10NonTrivialLi1EED1Ev(ptr addrspace(4) {{.*}}%PM.ascast)
// CHECK-NEXT: br label %wg_cf
// CHECK: wg_cf:
// CHECK: ret void
int main() {
  sycl::queue Q;
  Pod Captured{0};
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class Kernel>(
        sycl::range<1>{1}, sycl::range<1>{1}, [=](sycl::group<1> G) {
          // GLOBAL-DAG: @{{.*}}E6Scalar = internal addrspace(3) global i32 undef, align 4
          int Scalar = 1;
          // GLOBAL-DAG: @{{.*}}E1P = internal addrspace(3) global %struct.Pod undef, align 4
          Pod P{2};
          sycl::private_memory<NonTrivial> PM(G);
          // GLOBAL-DAG: @{{.*}}E3Ext = internal addrspace(3) global ptr addrspace(4) undef, align 8
          const NonTrivial &Ext = makeNonTrivial();
          makeNonTrivial();
          (void)Captured;
          // A work_group_static object declared without the static keyword at
          // work group scope is initialized with undef. Outside of
          // parallel_for_work_group it is zero-initialized like a static object
          // (see work_group_static.cpp).
          // GLOBAL-DAG: @{{.*}}E3WGS = internal addrspace(3) global %class.work_group_static zeroinitializer, align 4
          work_group_static<int> WGS;
          // GLOBAL-DAG: @{{.*}}E9StaticWGS = internal addrspace(3) global %class.work_group_static zeroinitializer, align 4
          static work_group_static<int> StaticWGS;
          static_cast<int &>(WGS) = 3;
          static_cast<int &>(StaticWGS) = 4;
        });
  });
  return 0;
}
