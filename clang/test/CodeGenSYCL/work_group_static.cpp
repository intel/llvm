// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=GLOBAL < %t.ll
// RUN: FileCheck %s --implicit-check-not=__cxa_atexit --implicit-check-not=_ZGV --implicit-check-not=llvm.global_ctors < %t.ll

// This test checks the code generated for objects of a type decorated with
// [[__sycl_detail__::wg_scope]], as used by sycl_ext_oneapi_work_group_static,
// at each scope allowed by the extension: namespace, class, block and lambda
// scope, with and without the static keyword. The objects are placed in local
// memory (addrspace(3)), are zero-initialized like static objects, and do not
// use guarded initialization or destructor registration.
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
  work_group_static &operator=(const T &value) noexcept {
    data = value;
    return *this;
  }

private:
  T data;
};

struct Pod {
  int I;
  float F;
};

// GLOBAL-DAG: @NsScalar = linkonce_odr addrspace(3) global %class.work_group_static zeroinitializer, align 4
work_group_static<int> NsScalar;
// GLOBAL-DAG: @_ZL13NsStaticArray = internal addrspace(3) global %class.work_group_static.[[#]] zeroinitializer, align 4
static work_group_static<int[16]> NsStaticArray;

struct ClassScope {
  static work_group_static<Pod> Member;
};
// GLOBAL-DAG: @_ZN10ClassScope6MemberE = linkonce_odr addrspace(3) global %class.work_group_static.[[#]] zeroinitializer, align 4
work_group_static<Pod> ClassScope::Member;

// CHECK-LABEL: define dso_local spir_func void @_Z11device_funci(
// CHECK: call {{.*}}@_ZN17work_group_staticIiEaSERKi({{.*}}addrspacecast (ptr addrspace(3) @_ZZ11device_funciE11BlockScalar to ptr addrspace(4)),
// CHECK: call {{.*}}@_ZN17work_group_staticIA4_iEcvRS0_Ev({{.*}}addrspacecast (ptr addrspace(3) @_ZZ11device_funciE16BlockStaticArray to ptr addrspace(4)))
SYCL_EXTERNAL void device_func(int V) {
  // GLOBAL-DAG: @_ZZ11device_funciE11BlockScalar = internal addrspace(3) global %class.work_group_static zeroinitializer, align 4
  work_group_static<int> BlockScalar;
  // GLOBAL-DAG: @_ZZ11device_funciE16BlockStaticArray = internal addrspace(3) global %class.work_group_static.[[#]] zeroinitializer, align 4
  static work_group_static<int[4]> BlockStaticArray;
  BlockScalar = V;
  static_cast<int(&)[4]>(BlockStaticArray)[0] = V;
}

// CHECK-LABEL: define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(
// CHECK: call {{.*}}@_ZN17work_group_staticIiEaSERKi({{.*}}addrspacecast (ptr addrspace(3) @NsScalar to ptr addrspace(4)),
// CHECK: call {{.*}}@_ZN17work_group_staticIA16_iEcvRS0_Ev({{.*}}addrspacecast (ptr addrspace(3) @_ZL13NsStaticArray to ptr addrspace(4)))
// CHECK: call {{.*}}@_ZN17work_group_staticI3PodEcvRS0_Ev({{.*}}addrspacecast (ptr addrspace(3) @_ZN10ClassScope6MemberE to ptr addrspace(4)))
// CHECK: call {{.*}}@_ZN17work_group_staticI3PodEcvRS0_Ev({{.*}}addrspacecast (ptr addrspace(3) @_ZZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEvE11LambdaScope to ptr addrspace(4)))
int main() {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class Kernel>([=]() {
      // GLOBAL-DAG: @_ZZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEvE11LambdaScope = internal addrspace(3) global %class.work_group_static.[[#]] zeroinitializer, align 4
      work_group_static<Pod> LambdaScope;
      NsScalar = 1;
      static_cast<int(&)[16]>(NsStaticArray)[0] = 2;
      static_cast<Pod &>(ClassScope::Member).I = 3;
      static_cast<Pod &>(LambdaScope).I = 4;
      device_func(5);
    });
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class KernelPFWG>(
        sycl::range<1>{1}, sycl::range<1>{1}, [=](sycl::group<1>) {
          // Sema marks the local variables of a parallel_for_work_group lambda
          // as work group scope variables, which must not override the static
          // semantics of work_group_static.
          // GLOBAL-DAG: @{{.*}}E6InPFWG = internal addrspace(3) global %class.work_group_static zeroinitializer, align 4
          work_group_static<int> InPFWG;
          InPFWG = 6;
        });
  });
  return 0;
}
