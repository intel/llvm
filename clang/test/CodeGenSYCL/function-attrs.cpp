// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes \
// RUN:  -triple spir64 -fexceptions -emit-llvm -fno-ident %s -o - | FileCheck %s --check-prefixes=CHECK,OGCG
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -fsycl-is-device -disable-llvm-passes -triple spir64 -fexceptions -emit-llvm -fno-ident %s -o - | FileCheck %s %}

int foo();

// The OGCG prefix checks the sycl-web specific !srcloc and SYCL module
// metadata, which ClangIR does not emit yet.

// CHECK: Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-LABEL: define linkonce_odr spir_func void @_Z3barv(
// CHECK-SAME: ) #[[ATTR1:[0-9]+]]
// OGCG-SAME: !srcloc [[META4:![0-9]+]] {
// CHECK:         [[A:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[A_ASCAST:%.*]] = addrspacecast ptr [[A]] to ptr addrspace(4)
// CHECK-NEXT:    [[CALL:%.*]] = call spir_func noundef i32 @_Z3foov() #[[ATTR2:[0-9]+]]
// CHECK-NEXT:    store i32 [[CALL]], ptr addrspace(4) [[A_ASCAST]], align 4
// CHECK-NEXT:    ret void
void bar() {
  int a = foo();
}

// CHECK: Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-LABEL: define linkonce_odr spir_func noundef i32 @_Z3foov(
// CHECK-SAME: ) #[[ATTR1]]
// OGCG-SAME: !srcloc [[META5:![0-9]+]] {
int foo() {
  return 1;
}

template <typename Name, typename... Ts>
void sycl_kernel_launch(Ts...) {}

template <typename Name, typename Func>
[[clang::sycl_kernel_entry_point(Name)]] void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([] { bar(); });
  return 0;
}
//.
// CHECK: attributes #[[ATTR0:[0-9]+]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="{{.*}}function-attrs.cpp" }
// CHECK: attributes #[[ATTR1]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
// CHECK: attributes #[[ATTR2]] = { convergent nounwind "uniform-work-group-size" }
//.
// OGCG: [[META0:![0-9]+]] = !{i32 1, !"sycl-device", i32 1}
// OGCG: [[META1:![0-9]+]] = !{i32 1, i32 2}
// OGCG: [[META2:![0-9]+]] = !{i32 4, i32 100000}
// OGCG: [[META3:![0-9]+]] = !{i32 1576}
// OGCG: [[META4]] = !{i32 1018}
// OGCG: [[META5]] = !{i32 1274}
//.
