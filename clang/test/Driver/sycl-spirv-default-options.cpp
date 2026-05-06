// Generate .bc file as SYCL device library file.
// REQUIRES: system-linux
//
// RUN: touch %t.devicelib.cpp
// RUN: %clang -cc1 -fsycl-is-device -emit-llvm-bc -o %t_1.devicelib.bc %t.devicelib.cpp
// RUN: %clang -cc1 -fsycl-is-device -emit-llvm-bc -o %t_2.devicelib.bc %t.devicelib.cpp
// RUN: %clang -cc1 -fsycl-is-device -emit-llvm-bc -o %t_3.devicelib.bc %t.devicelib.cpp

// Test for default llvm-spirv options

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=spir64-unknown-unknown -c %s -o %t_1.o
// RUN: clang-linker-wrapper --bitcode-library=spir64-unknown-unknown=%t_1.devicelib.bc \
// RUN:   "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" \
// RUN:   "--" "-o" "a.out" %t_1.o --dry-run 2>&1 | FileCheck  %s

// CHECK: llvm-spirv{{.*}}-spirv-debug-info-version=nonsemantic-shader-200
// CHECK-NOT: ocl-100
