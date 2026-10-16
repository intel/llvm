// Verify that we don't emit __sycl_registerlib_<unique identifier> nor __sycl_registerlib_ctor if -fembed-offload-object is not passed, as
// there is no device code in that case.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host --offload-new-driver -fsycl-unique-prefix=uid42 -emit-llvm %s -o - | FileCheck %s

// CHECK-NOT: __sycl_
