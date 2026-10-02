// Check the per-translation-unit __sycl_registerlib_<id> reference that SYCL
// host code generation emits under the new offload driver. The id is the
// translation unit's -fsycl-unique-prefix, which the driver generates.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host --offload-new-driver -fsycl-unique-prefix=uid42 -emit-llvm %s -o - | FileCheck %s

// Nothing is emitted without the new offload driver.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=uid42 -emit-llvm %s -o - | FileCheck %s --check-prefix=OLD

// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__sycl_registerlib_ctor, ptr null }]
// CHECK: declare void @__sycl_registerlib_uid42()
// CHECK: define internal void @__sycl_registerlib_ctor()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__sycl_registerlib_uid42()
// CHECK-NEXT: ret void

// OLD-NOT: __sycl_registerlib_

void foo() {}
