// Verify that compile-opts recorded in an OpenMP SPIR-V device image's
// offload metadata survive clang-linker-wrapper instead of being silently
// dropped when the image is re-containerized for the OpenMP JIT runtime.
// REQUIRES: system-linux
// REQUIRES: spirv-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o %t.bc
// RUN: %clang_cc1 %s -triple spirv64-intel -fopenmp-is-target-device -emit-obj -o %t.device.o
// RUN: llvm-offload-binary -o %t.bundle "--image=file=%t.device.o,triple=spirv64-intel,arch=generic,kind=openmp,compile-opts=-cl-opt-disable"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj -fembed-offload-object=%t.bundle -x ir %t.bc -o %t.host.o
// RUN: clang-linker-wrapper --dry-run --print-wrapped-module -host-triple=x86_64-unknown-linux-gnu %t.host.o -o %t.out --linker-path="/usr/bin/ld" > %t.ir 2>&1
// RUN: FileCheck %s --input-file=%t.ir

// CHECK: @.omp_offloading.device_image = {{.*}}compile-opts{{.*}}-cl-opt-disable

// Stub symbols required by the OpenMP offload descriptor registration code
// emitted by clang-linker-wrapper; avoids linking -lomp/-lomptarget/-lc.
extern "C" {
void __tgt_register_lib(void *) {}
void __tgt_unregister_lib(void *) {}
int atexit(void (*)(void)) { return 0; }
}

int main(int argc, char** argv) {
  return 0;
}
