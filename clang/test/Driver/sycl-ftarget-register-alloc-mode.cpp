// Test SYCL -ftarget-register-alloc-mode

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AUTO_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LARGE_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:small %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SMALL_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:default %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -Xs "-device pvc" %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -Xs "-device 0x0BD5" %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -Xs "-device 12.60.7" %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -Xs "-device pvc,mtl-s" %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:small,pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTIPLE_ARGS_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -ftarget-register-alloc-mode=pvc:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AUTO_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -ftarget-register-alloc-mode=pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LARGE_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -ftarget-register-alloc-mode=pvc:small %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SMALL_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -ftarget-register-alloc-mode=pvc:default %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_JIT %} %else %{ -check-prefix=AUTO_JIT %} %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -ftarget-register-alloc-mode=pvc:small,pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTIPLE_ARGS_JIT %s

// RUN: not %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=dg2:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD_DEVICE %s

// RUN: not %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:superlarge %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD_MODE %s

// RUN: not %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=dg2:superlarge %s 2>&1 \
// RUN:   | FileCheck -check-prefix=BAD_BOTH %s

// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen -Xs "-device bdw" \
// RUN:          %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_PVC %s

// TODO: Test fails on Windows due to improper temporary file creation given
//       the -device * value.  Re-enable when this is fixed.
// RUNx: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen -Xs "-device *" \
// RUNx:          %s 2>&1 \
// RUNx:   | FileCheck -check-prefix=NO_PVC %s

// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen -Xs "-device pvc:mtl-s" \
// RUN:          %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_PVC %s

// NO_PVC-NOT: -device_options
// NO_PVC-NOT: -ze-opt-large-register-file

// AUTO_AOT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch={{.*}},kind=sycl,compile-opts=-device_options pvc -ze-intel-enable-auto-large-GRF-mode{{.*}}"

// LARGE_AOT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=,kind=sycl,compile-opts=-device_options pvc -ze-opt-large-register-file"

// SMALL_AOT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=,kind=sycl,compile-opts=-device_options pvc -ze-intel-128-GRF-per-thread"

// DEFAULT_AOT-NOT: -device_options

// MULTIPLE_ARGS_AOT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=,kind=sycl,compile-opts=-device_options pvc -ze-intel-128-GRF-per-thread -device_options pvc -ze-opt-large-register-file"

// AUTO_JIT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch={{.*}},kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-enable-auto-large-GRF-mode"

// LARGE_JIT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch=,kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file"

// SMALL_JIT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch=,kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-128-GRF-per-thread"

// DEFAULT_JIT-NOT: -ftarget-register-alloc-mode=

// MULTIPLE_ARGS_JIT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch=,kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-128-GRF-per-thread -ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file"

// BAD_DEVICE: unsupported argument 'dg2:auto' to option '-ftarget-register-alloc-mode='
// BAD_MODE: unsupported argument 'pvc:superlarge' to option '-ftarget-register-alloc-mode='
// BAD_BOTH: unsupported argument 'dg2:superlarge' to option '-ftarget-register-alloc-mode='
