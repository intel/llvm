// RUN: not clang++ -O0 %s -w -o - -S -emit-llvm -fsycl -fsycl-device-only \
// RUN: -fsycl-targets=nvptx-unknown-unknown-syclmlir -nocudalib 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-NVPTX

// RUN: clang++ -O0 %s -o - -S -emit-llvm -fsycl -fsycl-device-only \
// RUN: -fsycl-targets=spir-unknown-unknown-syclmlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-32SPIR

// RUN: clang++ -O0 %s -w -o - -S -emit-llvm -fsycl -fsycl-device-only \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-64SPIR

#include <sycl/sycl.hpp>

// CHECK-NVPTX: Cannot lower SYCL target "nvptx-unknown-unknown-syclmlir" to LLVM

// CHECK-32SPIR: Using the 32-bits spir target may lead to errors when lowering the `sycl` dialect.

// CHECK-64SPIR-NOT: Cannot lower SYCL target

SYCL_EXTERNAL void foo() {}
