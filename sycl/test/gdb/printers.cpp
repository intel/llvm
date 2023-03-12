// RUN: %clangxx -fsyntax-only -fno-color-diagnostics -std=c++17 -I %sycl_include/sycl -I %sycl_include -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <sycl/buffer.hpp>
#include <sycl/detail/array.hpp>

typedef sycl::id<1> dummy_id;
typedef sycl::buffer<int> dummy_buffer;

// array must have common_array field

// CHECK: CXXRecordDecl {{.*}} class array definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced common_array

// buffer_plain must have impl field

// CHECK: CXXRecordDecl {{.*}} class buffer_plain definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced impl

// Check buffer inherits from buffer_plain

// CHECK: CXXRecordDecl {{.*}} class buffer definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: detail::buffer_plain':'sycl::detail::buffer_plain
