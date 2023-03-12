// RUN: %clangxx -fsyntax-only -fsycl -fno-color-diagnostics -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <sycl/group.hpp>
#include <sycl/id.hpp>

typedef sycl::private_memory<sycl::id<1>, 1> dummy;

// private_memory must have Val field of unique_ptr<T [], ...> type

// CHECK: CXXRecordDecl {{.*}} class private_memory definition
// CHECK: FieldDecl {{.*}} referenced Val {{.*}}:'unique_ptr<T[]>'
