// Test that clang-linker-wrapper correctly invokes clang-sycl-linker when --use-clang-sycl-linker is enabled
//
// REQUIRES: spirv-registered-target
//
// Create SYCL offload object file
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -c %s -o %t.o
//
// Test that clang-sycl-linker is invoked with basic options
// RUN: clang-linker-wrapper --dry-run --use-clang-sycl-linker \
// RUN:     --linker-path=/usr/bin/ld --host-triple=x86_64-unknown-linux-gnu \
// RUN:     %t.o -o %t.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-BASIC
// CHECK-BASIC: clang{{.*}}-fsycl --sycl-link
// CHECK-BASIC-SAME: -Xlinker --triple=spir64-unknown-unknown
//
// Test that --save-temps is forwarded
// RUN: clang-linker-wrapper --dry-run --use-clang-sycl-linker --save-temps \
// RUN:     --linker-path=/usr/bin/ld --host-triple=x86_64-unknown-linux-gnu \
// RUN:     %t.o -o %t.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SAVE-TEMPS
// CHECK-SAVE-TEMPS: clang{{.*}}-fsycl --sycl-link
// CHECK-SAVE-TEMPS-SAME: -Xlinker --save-temps
//
// Test that device compiler options are forwarded
// RUN: clang-linker-wrapper --dry-run --use-clang-sycl-linker \
// RUN:     --linker-path=/usr/bin/ld --host-triple=x86_64-unknown-linux-gnu \
// RUN:     --device-compiler=sycl:spir64-unknown-unknown=-my-backend-opt \
// RUN:     %t.o -o %t.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE-COMPILER
// CHECK-DEVICE-COMPILER: clang{{.*}}-fsycl --sycl-link
// CHECK-DEVICE-COMPILER-SAME: -Xlinker --device-compiler=sycl:spir64-unknown-unknown=-my-backend-opt
//
// Test that device linker options are forwarded
// RUN: clang-linker-wrapper --dry-run --use-clang-sycl-linker \
// RUN:     --linker-path=/usr/bin/ld --host-triple=x86_64-unknown-linux-gnu \
// RUN:     --device-linker=sycl:spir64-unknown-unknown=-my-linker-opt \
// RUN:     %t.o -o %t.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE-LINKER
// CHECK-DEVICE-LINKER: clang{{.*}}-fsycl --sycl-link
// CHECK-DEVICE-LINKER-SAME: -Xlinker --device-linker=sycl:spir64-unknown-unknown=-my-linker-opt

int foo() { return 42; }
