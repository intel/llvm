/// Verify same set of sycl-post-link options generated for old and new offloading model

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -### \
// RUN:          --no-offload-new-driver -Xdevice-post-link -O0 -fsycl-id-queries-range=int %s --sysroot=%S/Inputs/SYCL 2>&1 \
// RUN:   | FileCheck -check-prefix OPTIONS_POSTLINK_JIT_OLD %s
// OPTIONS_POSTLINK_JIT_OLD: sycl-post-link{{.*}} "-O2" "-device-globals" "-id-queries-range=int" "--device-lib-dir={{.*}}" "-properties" "-spec-const=native" "-emit-only-kernels-as-entry-points" "-emit-param-info" "-symbols" "-emit-exported-symbols" "-emit-imported-symbols" "-split-esimd" "-lower-esimd" "-O0"
//
// Ensure the driver forwards these options to clang-linker-wrapper.
//
// RUN: %clangxx %s -### -fsycl --offload-new-driver \
// RUN:   --sysroot=%S/Inputs/SYCL \
// RUN:   -fsycl-remove-unused-external-funcs \
// RUN:   -fsycl-device-code-split-esimd \
// RUN:   -fsycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD %s
// RUN: %clang_cl %s -### -fsycl --offload-new-driver \
// RUN:   /clang:--sysroot=%S/Inputs/SYCL \
// RUN:   -fsycl-remove-unused-external-funcs \
// RUN:   -fsycl-device-code-split-esimd \
// RUN:   -fsycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD %s
// OPTIONS_FORWARD: clang-linker-wrapper{{.*}} "-sycl-remove-unused-external-funcs" "-sycl-device-code-split-esimd" "-sycl-add-default-spec-consts-image"
//
// RUN: %clangxx %s -### -fsycl --offload-new-driver \
// RUN:   --sysroot=%S/Inputs/SYCL \
// RUN:   -fno-sycl-remove-unused-external-funcs \
// RUN:   -fno-sycl-device-code-split-esimd \
// RUN:   -fno-sycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD_NO %s
// RUN: %clang_cl %s -### -fsycl --offload-new-driver \
// RUN:   /clang:--sysroot=%S/Inputs/SYCL \
// RUN:   -fno-sycl-remove-unused-external-funcs \
// RUN:   -fno-sycl-device-code-split-esimd \
// RUN:   -fno-sycl-add-default-spec-consts-image \
// RUN:   2>&1 | FileCheck --check-prefix=OPTIONS_FORWARD_NO %s
// OPTIONS_FORWARD_NO: clang-linker-wrapper{{.*}} "-no-sycl-remove-unused-external-funcs" "-no-sycl-device-code-split-esimd" "-no-sycl-add-default-spec-consts-image"
