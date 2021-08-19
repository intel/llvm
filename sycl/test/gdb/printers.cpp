// RUN: %clangxx -c -fno-color-diagnostics -std=c++17 -I %sycl_include/sycl -I %sycl_include -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/array.hpp>

typedef cl::sycl::id<1> dummy_id;
typedef cl::sycl::buffer<int> dummy_buffer;

// array must have common_array field

// CHECK: CXXRecordDecl {{.*}} class array definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced common_array

// buffer must have impl field

// CHECK: CXXRecordDecl {{.*}} class buffer definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced impl
