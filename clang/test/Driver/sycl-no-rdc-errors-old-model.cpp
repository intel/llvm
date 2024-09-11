/// Tests driver errors for -no-sycl-rdc

// RUN: %clang -target %itanium_abi_triple -c %s -o %t.o
// RUN: %clang -target spir64_gen -emit-llvm -c %s -o %t
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,sycl-spir64_gen-unknown-unknown -input=%t -input=%t.o -output=%t.fat.o
// RUN: not %clang -### -fsycl --no-offload-new-driver -fno-sycl-rdc %t.fat.o 2>&1 | FileCheck -check-prefix=CHECK-ARCH %s

// CHECK-ARCH: error: linked binaries do not contain expected 'spir64-unknown-unknown' target; found targets: 'spir64_gen-unknown-unknown', this is not supported with '-fno-sycl-rdc'

// Some code so that we can create a binary out of this file.
void test_func(void) {
}
