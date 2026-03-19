// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/compile_time_kernel_info.hpp>

void foo(sycl::detail::compile_time_kernel_info_v1::CompileTimeKernelInfoTy) {}

// CHECK:       0 | struct sycl::detail::CompileTimeKernelInfoTy
// CHECK:       0 |   class sycl::detail::string_view Name
// CHECK-NEXT:  0 |     const char * str
// CHECK-NEXT:  8 |     size_t len
// CHECK-NEXT: 16 |   unsigned int NumParams
// CHECK-NEXT: 20 |   _Bool IsESIMD
// CHECK-NEXT: 24 |   class sycl::detail::string_view FileName
// CHECK-NEXT: 24 |     const char * str
// CHECK-NEXT: 32 |     size_t len
// CHECK-NEXT: 40 |   class sycl::detail::string_view FunctionName
// CHECK-NEXT: 40 |     const char * str
// CHECK-NEXT: 48 |     size_t len
// CHECK-NEXT: 56 |   unsigned int LineNumber
// CHECK-NEXT: 60 |   unsigned int ColumnNumber
// CHECK-NEXT: 64 |   int64_t KernelSize
// CHECK-NEXT: 72 |   ParamDescGetterT ParamDescGetter
// CHECK-NEXT: 80 |   _Bool HasSpecialCaptures
// CHECK-NEXT:    | [sizeof=88, dsize=81, align=8,
// CHECK-NEXT:    |  nvsize=81, nvalign=8]
