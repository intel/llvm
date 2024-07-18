/// Test for any deprecated options
// RUN: %clangxx -fsycl-explicit-simd %s -### 2>&1 | FileCheck %s -DOPTION=-fsycl-explicit-simd
// RUN: %clangxx -fno-sycl-explicit-simd %s -### 2>&1 | FileCheck %s -DOPTION=-fno-sycl-explicit-simd
// RUN: %clangxx -fsycl -fsycl-link-huge-device-code %s -### 2>&1 | FileCheck %s -DOPTION=-fsycl-link-huge-device-code
// RUN: %clangxx -fsycl -fno-sycl-link-huge-device-code %s -### 2>&1 | FileCheck %s -DOPTION=-fno-sycl-link-huge-device-code
// RUN: %clangxx -fsycl -fsycl-use-bitcode %s -### 2>&1 | FileCheck %s -DOPTION=-fsycl-use-bitcode
// RUN: %clangxx -fsycl -fno-sycl-use-bitcode %s -### 2>&1 | FileCheck %s -DOPTION=-fno-sycl-use-bitcode
// CHECK: option '[[OPTION]]' is deprecated and will be removed in a future release
