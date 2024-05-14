// Test SYCL -ftarget-register-alloc-mode

// RUN: %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AUTO_AOT %s

// RUN: %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LARGE_AOT %s

// RUN: %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:small %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SMALL_AOT %s

// RUN: %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:default %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT_AOT %s

// RUN: %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s

// RUN: %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:small,pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTIPLE_ARGS_AOT %s

// RUN: %clang -### -fsycl \
// RUN:   -ftarget-register-alloc-mode=pvc:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AUTO_JIT %s

// RUN: %clang -### -fsycl \
// RUN:   -ftarget-register-alloc-mode=pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LARGE_JIT %s

// RUN: %clang -### -fsycl \
// RUN:   -ftarget-register-alloc-mode=pvc:small %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SMALL_JIT %s

// RUN: %clang -### -fsycl \
// RUN:   -ftarget-register-alloc-mode=pvc:default %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT_JIT %s

// RUN: %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_JIT %} %else %{ -check-prefix=AUTO_JIT %} %s

// RUN: %clang -### -fsycl \
// RUN:   -ftarget-register-alloc-mode=pvc:small,pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTIPLE_ARGS_JIT %s

// RUN: not %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=dg2:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD_DEVICE %s

// RUN: not %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:superlarge %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD_MODE %s

// RUN: not %clang -### -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=dg2:superlarge %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD_BOTH %s

// AUTO_AOT: ocloc{{.*}} "-output"
// AUTO_AOT: -device_options
// AUTO_AOT: pvc
// AUTO_AOT: "-ze-intel-enable-auto-large-GRF-mode"

// LARGE_AOT: ocloc{{.*}} "-output"
// LARGE_AOT: -device_options
// LARGE_AOT: pvc
// LARGE_AOT: "-ze-opt-large-register-file"

// SMALL_AOT: ocloc{{.*}} "-output"
// SMALL_AOT: -device_options
// SMALL_AOT: pvc
// SMALL_AOT: "-ze-intel-128-GRF-per-thread"

// DEFAULT_AOT-NOT: -device_options

// MULTIPLE_ARGS_AOT: ocloc{{.*}} "-output"
// MULTIPLE_ARGS_AOT: -device_options
// MULTIPLE_ARGS_AOT: pvc
// MULTIPLE_ARGS_AOT: "-ze-intel-128-GRF-per-thread"
// MULTIPLE_ARGS_AOT: -device_options
// MULTIPLE_ARGS_AOT: pvc
// MULTIPLE_ARGS_AOT: "-ze-opt-large-register-file"

// AUTO_JIT: clang-offload-wrapper{{.*}} "-compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-enable-auto-large-GRF-mode"

// LARGE_JIT: clang-offload-wrapper{{.*}} "-compile-opts=-ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file"

// SMALL_JIT: clang-offload-wrapper{{.*}} "-compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-128-GRF-per-thread"

// DEFAULT_JIT-NOT: -ftarget-register-alloc-mode=

// MULTIPLE_ARGS_JIT: clang-offload-wrapper{{.*}} "-compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-128-GRF-per-thread -ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file"

// BAD_DEVICE: unsupported argument 'dg2:auto' to option '-ftarget-register-alloc-mode='
// BAD_MODE: unsupported argument 'pvc:superlarge' to option '-ftarget-register-alloc-mode='
// BAD_BOTH: unsupported argument 'dg2:superlarge' to option '-ftarget-register-alloc-mode='
