// Generate .bc file as SYCL device library file.
// REQUIRES: system-linux, libdevice
//
// RUN: touch %t.devicelib.bc

// Test for default llvm-spirv options

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-instrument-device-code \
// RUN:   -fsycl-targets=spir64-unknown-unknown -c %s -o %t.o
// RUN: clang-linker-wrapper --bitcode-library=spir64-unknown-unknown=%t.devicelib.bc \
// RUN:   "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" \
// RUN:   "--" "-o" "a.out" %t.o --dry-run 2>&1 | FileCheck  %s

// CHECK: llvm-spirv{{.*}}-spirv-debug-info-version=nonsemantic-shader-200
// CHECK-NOT: ocl-100
