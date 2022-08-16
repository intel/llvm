// RUN: %clangxx -fsycl-device-only -c -fno-color-diagnostics -Xclang -ast-dump %s -I %sycl_include -Wno-sycl-strict | FileCheck %s
// UNSUPPORTED: windows
#include <sycl/accessor.hpp>

typedef sycl::accessor<int, 1, sycl::access::mode::read> dummy;

// AccessorImplDevice must have MemRange and Offset fields

// CHECK: CXXRecordDecl {{.*}} class AccessorImplDevice definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced Offset
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced MemRange

// accessor.impl must be present and of AccessorImplDevice type

// CHECK: CXXRecordDecl {{.*}} class accessor definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced impl 'detail::AccessorImplDevice<AdjustedDim>'
