// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -femit-all-decls -o - | FileCheck %s

// This should not crash and we should not emit this declaration, even though
// we have 'emit-all-decls'.
// CHECK-NOT: define
void foo(void);
