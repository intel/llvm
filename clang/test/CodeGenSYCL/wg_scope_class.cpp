// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that variables whose type is decorated with the
// [[__sycl_detail__::wg_scope]] attribute (e.g. work_group_static) are placed
// in the local address space (addrspace(3)) when used from a regular (non
// hierarchical) kernel. These variables behave as if they had static storage
// duration, so they are zero initialized, including at block scope.

struct [[__sycl_detail__::wg_scope]] WG {
  int I;
};

template <typename T> struct [[__sycl_detail__::wg_scope]] WGTmpl {
  T I;
};

struct NotWG {
  int I;
};

struct Wrap {
  static WG Member;
};

WG Global;
static WG InternalGlobal;
WG Wrap::Member;
WGTmpl<float> TmplGlobal;
const NotWG ConstNotWGGlobal{};

// CHECK-DAG: @{{.*}}AutoWGS{{.*}} = internal addrspace(3) global %struct.WG zeroinitializer
// CHECK-DAG: @{{.*}}StaticWGS{{.*}} = internal addrspace(3) global %struct.WG zeroinitializer
// CHECK-DAG: @{{.*}}AutoTmpl{{.*}} = internal addrspace(3) global %struct.WGTmpl zeroinitializer
// CHECK-DAG: @Global = {{.*}}addrspace(3) global %struct.WG zeroinitializer
// CHECK-DAG: @_ZL14InternalGlobal = internal addrspace(3) global %struct.WG zeroinitializer
// CHECK-DAG: @_ZN4Wrap6MemberE = {{.*}}addrspace(3) global %struct.WG zeroinitializer
// CHECK-DAG: @TmplGlobal = {{.*}}addrspace(3) global %struct.WGTmpl zeroinitializer
// CHECK-DAG: @_ZL16ConstNotWGGlobal = internal addrspace(1) constant %struct.NotWG

// CHECK-LABEL: define {{.*}} @_Z8deviceFnv()
// CHECK-NOT: alloca %struct.WG
// CHECK: %NotWGLocal = alloca %struct.NotWG
// CHECK-NOT: alloca %struct.WG
// CHECK: ret
int deviceFn() {
  WG AutoWGS;
  static WG StaticWGS;
  WGTmpl<float> AutoTmpl;
  NotWG NotWGLocal;
  return AutoWGS.I + StaticWGS.I + (int)AutoTmpl.I + NotWGLocal.I + Global.I +
         InternalGlobal.I + Wrap::Member.I + (int)TmplGlobal.I +
         ConstNotWGGlobal.I;
}

// Work group scope variables must not use guarded initialization.
// CHECK-NOT: __cxa_guard

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &K) {
  K();
}

int main() {
  kernel_single_task<class Kernel>([]() { deviceFn(); });
  return 0;
}
