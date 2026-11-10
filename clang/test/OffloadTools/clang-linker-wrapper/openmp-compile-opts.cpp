// Verify that compile-opts/link-opts recorded in an OpenMP SPIR-V device
// image's offload metadata survive clang-linker-wrapper instead of being
// silently dropped when the image is re-containerized for the OpenMP JIT
// runtime.
// REQUIRES: system-linux
// REQUIRES: spirv-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o %t.bc
// RUN: %clang_cc1 %s -triple spirv64-intel -fopenmp-is-target-device -emit-obj -o %t.device.o
// RUN: llvm-offload-binary -o %t.bundle "--image=file=%t.device.o,triple=spirv64-intel,arch=generic,kind=openmp,compile-opts=-cl-opt-disable,link-opts=-cl-take-global-address"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -fembed-offload-object=%t.bundle -x ir %t.bc -o %t.host.o
// RUN: clang-linker-wrapper --dry-run --print-wrapped-module -host-triple=x86_64-unknown-linux-gnu %t.host.o -o %t.out --linker-path="/usr/bin/ld" > %t.ir 2>&1
//
// The embedded strings are serialized through a string-table builder that
// may reorder them relative to each other, so each is checked with a
// separate FileCheck run rather than relying on their relative order.
// RUN: FileCheck %s --input-file=%t.ir --check-prefix=COMPILE-OPTS
// RUN: FileCheck %s --input-file=%t.ir --check-prefix=LINK-OPTS

// COMPILE-OPTS: @.omp_offloading.device_image = {{.*}}compile-opts{{.*}}-cl-opt-disable
// LINK-OPTS: @.omp_offloading.device_image = {{.*}}link-opts{{.*}}-cl-take-global-address
