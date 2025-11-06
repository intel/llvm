// RUN: %clangxx -fsycl -c -fno-color-diagnostics -Xclang -fdump-record-layouts %s -o %t.out | FileCheck %s
// REQUIRES: linux
// UNSUPPORTED: libcxx

// clang-format off

#include <sycl/detail/compile_time_kernel_info.hpp>

void foo(sycl::detail::compile_time_kernel_info_v1::CompileTimeKernelInfoTy) {}

// CHECK:       0 | struct sycl::detail::CompileTimeKernelInfoTy
// CHECK:       0 |   class sycl::detail::string_view Name
// CHECK-NEXT:  0 |     const char * str
// CHECK-NEXT:  8 |   unsigned int NumParams
// CHECK-NEXT: 12 |   _Bool IsESIMD
// CHECK-NEXT: 16 |   class sycl::detail::string_view FileName
// CHECK-NEXT: 16 |     const char * str
// CHECK-NEXT: 24 |   class sycl::detail::string_view FunctionName
// CHECK-NEXT: 24 |     const char * str
// CHECK-NEXT: 32 |   unsigned int LineNumber
// CHECK-NEXT: 36 |   unsigned int ColumnNumber
// CHECK-NEXT: 40 |   int64_t KernelSize
// CHECK-NEXT: 48 |   ParamDescGetterT ParamDescGetter
// CHECK-NEXT: 56 |   _Bool HasSpecialCaptures
// CHECK-NEXT:    | [sizeof=64, dsize=57, align=8,
// CHECK-NEXT:    |  nvsize=57, nvalign=8]
