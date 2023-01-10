// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsycl-int-footer=%t.h %s -emit-llvm -o %t.ll
// RUN: FileCheck -input-file=%t.h %s --allow-empty

// CHECK-NOT: #include <sycl/detail/defines_elementary.hpp>
// CHECK-NOT: #include <sycl/detail/spec_const_integration.hpp>
