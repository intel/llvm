// RUN: %clangxx -c -fsycl -fno-color-diagnostics -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>

typedef cl::sycl::private_memory<cl::sycl::id<1>, 1> dummy;

// private_memory must have Val field of unique_ptr<T [], ...> type

// CHECK: CXXRecordDecl {{.*}} class private_memory definition
// CHECK: FieldDecl {{.*}} referenced Val {{.*}}:'unique_ptr<T []>'
