/// Verify same set of sycl-post-link options generated for old and new offloading model
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -### \
// RUN:          --no-offload-new-driver -Xdevice-post-link -O0 %s --sysroot=%S/Inputs/SYCL 2>&1 \
// RUN:   | FileCheck -check-prefix OPTIONS_POSTLINK_JIT_OLD %s
// OPTIONS_POSTLINK_JIT_OLD: sycl-post-link{{.*}} "-O2" "-device-globals" "--device-lib-dir={{.*}}" "-properties" "-spec-const=native" "-split=auto" "-emit-only-kernels-as-entry-points" "-emit-param-info" "-symbols" "-emit-exported-symbols" "-emit-imported-symbols" "-split-esimd" "-lower-esimd" "-O0"
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
//
// Run clang-linker-wrapper test for generating SYCLBIN files.
//
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -syclbin=executable -sycl-device-libraries=%t.devicelib.o \
// RUN:   --sycl-post-link-options="-O2 -device-globals -O0" \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck --check-prefix OPTIONS_POSTLINK_JIT_NEW_SYCLBIN %s
// OPTIONS_POSTLINK_JIT_NEW_SYCLBIN: sycl-post-link{{.*}} -spec-const=native -properties -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-kernel-names -emit-exported-symbols -emit-imported-symbols -split-esimd -lower-esimd -O2 -device-globals -O0
//
// Ensure driver forwards these triple based options to clang-linker-wrapper.
// 
// RUN: %clangxx %s -### -fsycl --offload-new-driver \
// RUN:   -fsycl-remove-unused-external-funcs \
// RUN:   -fsycl-device-code-split-esimd \
// RUN:   -fsycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD %s
// RUN: %clang_cl %s -### -fsycl --offload-new-driver \
// RUN:   -fsycl-remove-unused-external-funcs \
// RUN:   -fsycl-device-code-split-esimd \
// RUN:   -fsycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD %s
// OPTIONS_FORWARD: clang-linker-wrapper{{.*}} "-sycl-remove-unused-external-funcs" "-sycl-device-code-split-esimd" "-sycl-add-default-spec-consts-image"
//
// RUN: %clangxx %s -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-remove-unused-external-funcs \
// RUN:   -fno-sycl-device-code-split-esimd \
// RUN:   -fno-sycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD_NO %s
// RUN: %clang_cl %s -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-remove-unused-external-funcs \
// RUN:   -fno-sycl-device-code-split-esimd \
// RUN:   -fno-sycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD_NO %s
// OPTIONS_FORWARD_NO: clang-linker-wrapper{{.*}} "-no-sycl-remove-unused-external-funcs" "-no-sycl-device-code-split-esimd" "-no-sycl-add-default-spec-consts-image"
//
// Check -no-sycl-remove-unused-external-funcs option disables emitting
// -emit-only-kernels-as-entry-points in sycl-post-link.
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -sycl-device-libraries=%t.devicelib.o \
// RUN:   -no-sycl-remove-unused-external-funcs \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck --check-prefix OPTIONS_NO_EMIT_ONLY_KERNELS %s
// OPTIONS_NO_EMIT_ONLY_KERNELS: sycl-post-link{{.*}} -spec-const=native -properties -split=auto -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -split-esimd -lower-esimd
//
// Check -no-sycl-device-code-split-esimd option disables emitting
// -split-esimd in sycl-post-link.
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -sycl-device-libraries=%t.devicelib.o \
// RUN:   -no-sycl-device-code-split-esimd \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck --check-prefix OPTIONS_NO_SPLIT_ESIMD %s
// OPTIONS_NO_SPLIT_ESIMD: sycl-post-link{{.*}} -spec-const=native -properties -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -lower-esimd
// 
// Generate AOT .o file as linker wrapper input.
//
// RUN: %clang %s -fsycl -fsycl-targets=spir64_gen-unknown-unknown -c --offload-new-driver -o %t_aot.o
//
// Generate AOT .o file as SYCL device library file.
//
// RUN: touch %t.devicelib.cpp
// RUN: %clang %t.devicelib.cpp -fsycl -fsycl-targets=spir64_gen-unknown-unknown -c --offload-new-driver -o %t.devicelib_aot.o
//
// Check -sycl-add-default-spec-consts-image option enables emitting
// -generate-device-image-default-spec-consts in sycl-post-link.
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -sycl-device-libraries=%t.devicelib_aot.o \
// RUN:   -sycl-add-default-spec-consts-image \
// RUN:   --linker-path=/usr/bin/ld %t_aot.o -o a.out 2>&1 | FileCheck --check-prefix OPTIONS_DEFAULT_SPEC_CONSTS %s
// OPTIONS_DEFAULT_SPEC_CONSTS: sycl-post-link{{.*}} -spec-const=emulation -properties -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -emit-imported-symbols -split-esimd -lower-esimd -generate-device-image-default-spec-consts
