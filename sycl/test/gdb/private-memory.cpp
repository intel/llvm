// RUN: %clangxx -fsyntax-only -fsycl -fno-color-diagnostics -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <sycl/group.hpp>
#include <sycl/id.hpp>

typedef sycl::private_memory<sycl::id<1>, 1> dummy;

// private_memory must have Val field of unique_ptr<T [], ...> type

// CHECK: CXXRecordDecl {{.*}} class private_memory definition
// -FieldDecl 0x7f7f946e4a58 <line:94:3, col:24> col:24 referenced Val 'std::unique_ptr<T[]>'
// CHECK: FieldDecl {{.*}} referenced Val {{.*}}'std::unique_ptr<T[]>'
