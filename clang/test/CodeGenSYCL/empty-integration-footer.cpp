// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s --allow-empty

// CHECK-NOT: #include <CL/sycl/detail/defines_elementary.hpp>
// CHECK-NOT: #include <CL/sycl/detail/spec_const_integration.hpp>
