// RUN: %clangxx -c -fno-color-diagnostics -std=c++17 -I %sycl_include/sycl -I %sycl_include -Xclang -ast-dump %s | FileCheck %s
// UNSUPPORTED: windows
#include <CL/sycl/accessor.hpp>

typedef cl::sycl::accessor<int, 1, cl::sycl::access::mode::read> dummy;

// AccessorImplHost must have MMemoryRange, MOffset and MData fields

// CHECK: CXXRecordDecl {{.*}} class AccessorImplHost definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced MOffset
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced MMemoryRange
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced MData

// accessor.impl must be present and of shared_ptr<AccessorImplHost> type

// CHECK: CXXRecordDecl {{.*}} class AccessorBaseHost definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced impl {{.*}}:'std::shared_ptr<sycl::detail::AccessorImplHost>'

// LocalAccessorImplHost must have MSize and MMem fields

// CHECK: CXXRecordDecl {{.*}} class LocalAccessorImplHost definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced MSize
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: FieldDecl {{.*}} referenced MMem

// CHECK: CXXRecordDecl {{.*}} class accessor definition
// CHECK-NOT: CXXRecordDecl {{.*}} definition
// CHECK: public {{.*}}:'sycl::detail::AccessorBaseHost'
