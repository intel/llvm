// RUN: %clangxx -fsycl-device-only -c -fno-color-diagnostics -Xclang -ast-dump %s -I %sycl_include -Wno-sycl-strict | FileCheck %s
// UNSUPPORTED: windows
#include <sycl/group.hpp>
#include <sycl/id.hpp>

typedef sycl::private_memory<sycl::id<1>, 1> dummy;

// private_memory must have Val field of T type

// CHECK: CXXRecordDecl {{.*}} class private_memory definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced Val 'T'
