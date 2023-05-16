// REQUIRES: x86-registered-target

// This test check that clang-offload-bundler unbundles objects bundled with
// previous DPC++ toolchain version. The images created by old toolchain used
// target triples with `sycldevice` environment component, which is deprecated.
// Here we test the support for binaries created with old toolchain and
// deprecated triple.


// RUN: %clang -target %itanium_abi_triple -c %s -o %t.o
// RUN: %clang -target spir64 -emit-llvm -c %s -o %t

// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,sycl-spir64-unknown-unknown-sycldevice -input=%t.o -input=%t -output=%t.fat.o
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,sycl-spir64-unknown-unknown -output=%t.host -output=%t.device -input=%t.fat.o -unbundle
// RUN: diff %t %t.device

// Some code so that we can create a binary out of this file.
int A = 0;
void test_func(void) {
  ++A;
}
