// REQUIRES: x86-registered-target

// Check that clang-offload-wrapper defines the __sycl_registerlib_<id>
// functions referenced by -kind=host inputs, including archive members, that
// are not bundled or have device code for a wrapped target.

// RUN: echo 'Content of device file' > %t.dev
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Xclang -fsycl-is-host -Xclang -fsycl-unique-prefix=plain -Xclang -fembed-offload-object=%t.dev %s -o %t.plain.o
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Xclang -fsycl-is-host -Xclang -fsycl-unique-prefix=jit -Xclang -fembed-offload-object=%t.dev %s -o %t.jit.host.o
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Xclang -fsycl-is-host -Xclang -fsycl-unique-prefix=gen -Xclang -fembed-offload-object=%t.dev %s -o %t.gen.host.o
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown -input=%t.jit.host.o -input=%t.dev -output=%t.jit.o
// RUN: clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-pvc -input=%t.gen.host.o -input=%t.dev -output=%t.gen.o
// RUN: rm -f %t.a && llvm-ar rc %t.a %t.jit.o %t.gen.o
// RUN: echo -e '[Code|Symbols|Properties]\n%t.dev||' > %t.table

// RUN: clang-offload-wrapper -o %t.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=spir64 %t.dev -kind=host %t.plain.o -kind=host %t.a
// RUN: llvm-dis %t.bc -o - | FileCheck %s --implicit-check-not=__sycl_registerlib_gen
// RUN: clang-offload-wrapper -o %t.gen.bc -host=x86_64-unknown-linux-gnu -kind=sycl -target=spir64_gen -batch %t.table -kind=host %t.a
// RUN: llvm-dis %t.gen.bc -o - | FileCheck %s --check-prefix=GEN --implicit-check-not=__sycl_registerlib_jit --implicit-check-not=__sycl_registerlib_plain

// CHECK-DAG: define weak void @__sycl_registerlib_plain() comdat
// CHECK-DAG: define weak void @__sycl_registerlib_jit() comdat
// GEN: define weak void @__sycl_registerlib_gen() comdat

void foo() {}
