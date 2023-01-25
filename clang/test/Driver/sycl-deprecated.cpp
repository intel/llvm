/// Test for any deprecated options
// RUN: %clangxx -fsycl-explicit-simd %s -### 2>&1 | FileCheck %s -DOPTION=-fsycl-explicit-simd
// RUN: %clangxx -fno-sycl-explicit-simd %s -### 2>&1 | FileCheck %s -DOPTION=-fno-sycl-explicit-simd
// CHECK: option '[[OPTION]]' is deprecated and will be removed in a future release

// RUN: %clangxx -fsycl -sycl-std=1.2.1 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ARG -DOPTION=-sycl-std= \
// RUN:    -DARGUMENT=1.2.1
// RUN: %clangxx -fsycl -sycl-std=121 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ARG -DOPTION=-sycl-std= \
// RUN:     -DARGUMENT=121
// RUN: %clangxx -fsycl -sycl-std=sycl-1.2.1 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ARG -DOPTION=-sycl-std= \
// RUN:     -DARGUMENT=sycl-1.2.1
// RUN: %clangxx -fsycl -sycl-std=2017 %s -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=CHECK-ARG -DOPTION=-sycl-std= \
// RUN:     -DARGUMENT=2017
// CHECK-ARG: argument '[[ARGUMENT]]' for option '[[OPTION]]' is deprecated and will be removed in a future release
