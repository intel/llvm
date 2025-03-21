// RUN: %clangxx -fsycl -fsycl-host-compiler=cl %s \
// RUN:     -fsycl-host-compiler-options="/std:c++17 /Zc:__cplusplus" -c \
// RUN:     -o %t.out | FileCheck %s
// REQUIRES: windows

#include <sycl/sycl.hpp>

// CHECK: '__SYCL_COMPILER_VERSION': name was marked as #pragma deprecated
#if __SYCL_COMPILER_VERSION >= 2024
#endif
