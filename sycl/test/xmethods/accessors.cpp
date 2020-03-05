// RUN: %clangxx -c -fno-color-diagnostics -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <CL/sycl/accessor.hpp>

typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::read> dummy;

// AccessorBaseHost must have getOffset, getMemoryRange, and getPtr methods

// CHECK: CXXRecordDecl {{.*}} class AccessorBaseHost definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: CXXMethodDecl {{.*}} getOffset 'id<3> &()'
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: CXXMethodDecl {{.*}} getMemoryRange 'range<3> &()'
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: CXXMethodDecl {{.*}} getPtr 'void *()'
