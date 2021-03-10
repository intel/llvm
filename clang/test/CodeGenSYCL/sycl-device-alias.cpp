// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// Test that aliasing does not force an unused entity to be emitted

// CHECK-NOT: define {{.*}}spir_func void @unused_func()
extern "C" void unused_func() {}
// CHECK-NOT: @unused_aliaser
extern "C" void unused_aliaser() __attribute__((alias("unused_func")));
// CHECK-NOT: @unused_int
int unused_int = 3;
// CHECK-NOT: @alias_unused_int
extern int alias_unused_int __attribute__((alias("unused_int")));

// CHECK-DAG: define {{.*}}spir_func void @used_func()
extern "C" void used_func() {}
// CHECK-DAG: @aliaser = {{.*}}alias void (), void ()* @used_func
extern "C" void aliaser() __attribute__((alias("used_func")));

// CHECK-DAG: define {{.*}}spir_func void @func()
extern "C" void func() {}
// CHECK-DAG: @used_aliaser = {{.*}}alias void (), void ()* @func
extern "C" void used_aliaser() __attribute__((alias("func")));

// CHECK-DAG: @used_int = {{.*}}addrspace(1) constant i32 5, align 4
extern "C" const int used_int = 5;
// CHECK-DAG: @alias_used_int = {{.*}}alias i32, i32 addrspace(1)* @used_int
extern "C" const int alias_used_int __attribute__((alias("used_int")));
// CHECK-DAG: @vint = {{.*}}addrspace(1) constant i32 7, align 4
extern "C" const int vint = 7;
// CHECK-DAG: @used_alias_used_int = {{.*}}alias i32, i32 addrspace(1)* @vint
extern "C" const int used_alias_used_int __attribute__((alias("vint")));

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}bar{{.*}}
void bar(const int &i) {}

// CHECK-DAG: define {{.*}}spir_func void @{{.*}}foo{{.*}}
void __attribute__((sycl_device)) foo() {
  // CHECK-DAG: call spir_func void @{{.*}}bar{{.*}}@used_int
  bar(used_int);
  // CHECK-DAG: call spir_func void @{{.*}}bar{{.*}}@used_alias_used_int
  bar(used_alias_used_int);
  // CHECK-DAG: call spir_func void @used_func()
  used_func();
  // CHECK-DAG: call spir_func void @used_aliaser()
  used_aliaser();
}
