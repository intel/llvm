// RUN: %clangxx -c -fno-color-diagnostics -I %sycl_include -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <CL/sycl/accessor.hpp>

typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::read> dummy;

// AccessorBaseHost must have getOffset, getMemoryRange, and getPtr methods

// CHECK: CXXRecordDecl {{.*}} class AccessorBaseHost definition
// CHECK: CXXMethodDecl {{.*}} getOffset 'const id<3> &() const'
// CHECK: CXXMethodDecl {{.*}} getMemoryRange 'const range<3> &() const'
// CHECK: CXXMethodDecl {{.*}} getPtr 'void *() const'
