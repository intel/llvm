/// Check that --sycl-host-compiler-arg values are passed to the clang command
/// that compiles the wrapper.
// REQUIRES: system-linux, x86-registered-target

// RUN: %clang %s -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver --no-offloadlib -fno-sycl-instrument-device-code -o %t.o
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/ld -o /dev/null %t.o \
// RUN:   --sycl-host-compiler-arg=-march=skylake --sycl-host-compiler-arg=-mcmodel=large 2>&1 \
// RUN:   | FileCheck %s
// CHECK: offload-wrapper: output: [[WRAPPED:.*\.bc]]
// CHECK: clang{{.*}} --target=x86_64-unknown-linux-gnu -march=skylake -mcmodel=large -c -o {{.*}}.o [[WRAPPED]]

void foo() {}
