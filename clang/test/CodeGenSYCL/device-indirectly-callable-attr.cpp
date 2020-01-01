// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

void helper() {}

[[intel::device_indirectly_callable]]
void foo() {
  helper();
}

// CHECK: define spir_func void @{{.*foo.*}}() #[[ATTRS_FOO:[0-9]+]]
// CHECK: call spir_func void @{{.*helper.*}}()
//
// CHECK: define spir_func void @{{.*helper.*}}() #[[ATTRS_HELPER:[0-9]+]]
//
// CHECK: attributes #[[ATTRS_FOO]] = { {{.*}} "referenced-indirectly"
// CHECK-NOT: attributes #[[ATTRS_HELPER]] = { {{.*}} "referenced-indirectly"
