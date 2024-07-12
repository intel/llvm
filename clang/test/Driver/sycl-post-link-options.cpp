// REQUIRES: system-linux
/// Verify same set of sycl-post-link options generated for old and new offloading model
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -### \
// RUN:          --no-offload-new-driver -Xdevice-post-link -O0 %s 2>&1 \
// RUN:   | FileCheck -check-prefix OPTIONS_POSTLINK_JIT_OLD %s
// OPTIONS_POSTLINK_JIT_OLD: sycl-post-link{{.*}} "-O2" "-device-globals" "-properties" "-spec-const=native" "-split=auto" "-emit-only-kernels-as-entry-points" "-emit-param-info" "-symbols" "-emit-exported-symbols" "-emit-imported-symbols" "-split-esimd" "-lower-esimd" "-O0"
//
// Generate .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t.o
//
// Generate .o file as SYCL device library file.
//
// RUN: touch %t.devicelib.cpp
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver -o %t.devicelib.o
//
// Run clang-linker-wrapper test
//
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -sycl-device-libraries=%t.devicelib.o \
// RUN:   --sycl-post-link-options="-O2 -device-globals -O0" \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck --check-prefix OPTIONS_POSTLINK_JIT_NEW %s
// OPTIONS_POSTLINK_JIT_NEW: sycl-post-link{{.*}} -spec-const=native -properties -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -split-esimd -lower-esimd -O2 -device-globals -O0
