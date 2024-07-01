// REQUIRES: system-linux
/// Verify same set of sycl-post-link options generated for old and new offloading model
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -### \
// RUN:          -Xdevice-post-link -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix OPTIONS_POSTLINK_JIT_OLD %s
// OPTIONS_POSTLINK_JIT_OLD: sycl-post-link{{.*}} "-O2" "-device-globals" "-properties" "-spec-const=native" "-split=auto" "-emit-only-kernels-as-entry-points" "-emit-param-info" "-symbols" "-emit-exported-symbols" "-emit-imported-symbols" "-split-esimd" "-lower-esimd" "-O0"
//
// Generate .o file as linker wrapper input.
//
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: clang-offload-packager -o %t.out --image=file=%t.elf.o,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
//
// Generate .o file as SYCL device library file.
//
// RUN: echo '' > %t.devicelib.cpp
// RUN: %clang -cc1 %t.devicelib.cpp -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device -emit-llvm-bc -o %t.devicelib.bc
// RUN: clang-offload-packager -o %t.devicelib.out --image=file=%t.devicelib.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.devicelib.o \
// RUN:   -fembed-offload-object=%t.devicelib.out
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -sycl-device-library-location= -sycl-device-libraries=%t.devicelib.o \
// RUN:   --sycl-post-link-options="-O2 -device-globals -properties -O0" \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck --check-prefix OPTIONS_POSTLINK_JIT_NEW %s
// OPTIONS_POSTLINK_JIT_NEW: sycl-post-link{{.*}} -spec-const=native -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -split-esimd -lower-esimd -O2 -device-globals -properties -O0
