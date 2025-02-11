// Generate .o file as SYCL device library file.
//
// RUN: touch %t.devicelib.cpp
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t_1.devicelib.o
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64_gen-unknown-unknown -c --offload-new-driver -o %t_2.devicelib.o
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown -c --offload-new-driver -o %t_3.devicelib.o

// Test for default llvm-spirv options

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=spir64-unknown-unknown -c %s -o %t_1.o
// RUN: clang-linker-wrapper -sycl-device-libraries=%t_1.devicelib.o \
// RUN:   "--host-triple=x86_64-unknown-linux-gnu" "--linker-path=/usr/bin/ld" \
// RUN:   "--" "-o" "a.out" %t_1.o --dry-run 2>&1 | FileCheck  %s

// CHECK: llvm-spirv{{.*}}-spirv-debug-info-version=nonsemantic-shader-200
// CHECK-NOT: ocl-100
