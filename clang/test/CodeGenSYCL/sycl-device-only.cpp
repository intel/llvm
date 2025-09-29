// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECKD
// RUN: %clang_cc1 -fsycl-is-host -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECKH
// Test code generation for sycl_device_only attribute.

// Verify that the device overload is used on device.
//
// CHECK-LABEL: _Z3fooi
// CHECKH: %add = add nsw i32 %0, 10
// CHECKD: %add = add nsw i32 %0, 20
int foo(int a) { return a + 10; }
__attribute__((sycl_device_only)) int foo(int a) { return a + 20; }

// Use a `sycl_device` function as entry point
__attribute__((sycl_device)) int bar(int b) { return foo(b); }

// Verify that the order of declaration doesn't change the behavior.
//
// CHECK-LABEL: _Z3fooswapi
// CHECKH: %add = add nsw i32 %0, 10
// CHECKD: %add = add nsw i32 %0, 20
__attribute__((sycl_device_only)) int fooswap(int a) { return a + 20; }
int fooswap(int a) { return a + 10; }

// Use a `sycl_device` function as entry point.
__attribute__((sycl_device)) int barswap(int b) { return fooswap(b); }

// Verify that in extern C the attribute enables mangling.
extern "C" {
// CHECK-LABEL: _Z3fooci
// CHECKH: %add = add nsw i32 %0, 10
// CHECKD: %add = add nsw i32 %0, 20
int fooc(int a) { return a + 10; }
__attribute__((sycl_device_only)) int fooc(int a) { return a + 20; }

// Use a `sycl_device` function as entry point.
__attribute__((sycl_device)) int barc(int b) { return fooc(b); }
}
