// RUN: %clangxx -fsycl-device-only -c -fno-color-diagnostics -Xclang -ast-dump %s -I %sycl_include -Wno-sycl-strict | FileCheck %s
// UNSUPPORTED: windows
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>

typedef cl::sycl::private_memory<cl::sycl::id<1>, 1> dummy;

// private_memory must have Val field of T type

// CHECK: CXXRecordDecl {{.*}} class private_memory definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced Val 'T'
