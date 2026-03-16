// UNSUPPORTED: system-windows

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// RUN: %clang -cc1 %s -triple spirv64-unknown-unknown -emit-llvm-bc -o %t.spirv.bc
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=SPIRV-LINK