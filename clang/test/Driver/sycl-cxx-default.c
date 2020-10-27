/// When -fsycl is used, C++ source is the default
// REQUIRES: clang-driver

// RUN: %clang -c -fsycl %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX_TYPE_CHECK %s
// RUN: %clangxx -c -fsycl %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX_TYPE_CHECK %s
// RUN: %clang_cl -c -fsycl %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX_TYPE_CHECK %s
// RUN: %clang_cl -c -fsycl /TC %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX_TYPE_CHECK %s
// RUN: %clang_cl -c -fsycl /Tc%s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CXX_TYPE_CHECK %s
// CXX_TYPE_CHECK: "-x" "c++"
// CXX_TYPE_CHECK-NOT: "-x" "c"

// RUN: %clang -c -fsycl -std=c99 %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=C_SYCL_ERROR_CHECK %s
// C_SYCL_ERROR_CHECK: error: invalid argument '-std=c99' not allowed with '-fsycl
