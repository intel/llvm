// Test SYCL -ftarget-register-alloc-mode

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AUTO_AOT %s -DDEVICE=pvc

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LARGE_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:small %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SMALL_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:default %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s -DDEVICE=pvc

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s -DDEVICE=pvc

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s -DDEVICE=pvc

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s -DDEVICE=pvc

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc,intel_gpu_mtl_s -Xs "-device pvc,mtl-s" %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_AOT %} %else %{ -check-prefix=AUTO_AOT %} %s -DDEVICE=pvc

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:small,pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTIPLE_ARGS_AOT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -ftarget-register-alloc-mode=pvc:auto %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AUTO_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -ftarget-register-alloc-mode=pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=LARGE_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -ftarget-register-alloc-mode=pvc:small %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SMALL_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -ftarget-register-alloc-mode=pvc:default %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT_JIT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL %s 2>&1 \
// RUN:   | FileCheck %if system-windows %{ -check-prefix=DEFAULT_JIT %} %else %{ -check-prefix=AUTO_JIT %} %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -ftarget-register-alloc-mode=pvc:small,pvc:large %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTIPLE_ARGS_JIT %s

// TODO: consider the following cases
// rUN: not %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// rUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=dg2:auto %s 2>&1 \
// rUN:   | FileCheck -check-prefix=BAD_DEVICE %s

// rUN: not %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// rUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=pvc:superlarge %s 2>&1 \
// rUN:   | FileCheck -check-prefix=BAD_MODE %s

// rUN: not %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// rUN:    -fsycl-targets=spir64_gen -ftarget-register-alloc-mode=dg2:superlarge %s 2>&1 \
// rUN:   | FileCheck -check-prefix=BAD_BOTH %s

// RUN: %clangxx -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl-targets=intel_gpu_bdw %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NO_PVC %s

// rUN: %clangxx -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl-targets=intel_gpu_pvc,intel_gpu_mtl_s \
// rUN:          %s 2>&1 \
// rUN:   | FileCheck -check-prefix=NO_PVC %s

// NO_PVC-NOT: -device_options
// NO_PVC-NOT: -ze-opt-large-register-file

// AUTO_AOT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch={{.*}},kind=sycl,compile-opts=-device_options [[DEVICE]] -ze-intel-enable-auto-large-GRF-mode{{.*}}"

// LARGE_AOT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=pvc,kind=sycl,compile-opts=-device_options pvc -ze-opt-large-register-file"

// SMALL_AOT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=pvc,kind=sycl,compile-opts=-device_options pvc -ze-intel-128-GRF-per-thread"

// DEFAULT_AOT-NOT: -device_options

// MULTIPLE_ARGS_AOT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch=pvc,kind=sycl,compile-opts=-device_options pvc -ze-intel-128-GRF-per-thread -device_options pvc -ze-opt-large-register-file"

// AUTO_JIT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch={{.*}},kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-enable-auto-large-GRF-mode"

// LARGE_JIT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch=generic,kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file"

// SMALL_JIT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch=generic,kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-128-GRF-per-thread"

// DEFAULT_JIT-NOT: -ftarget-register-alloc-mode=

// MULTIPLE_ARGS_JIT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch=generic,kind=sycl,compile-opts=-ftarget-register-alloc-mode=pvc:-ze-intel-128-GRF-per-thread -ftarget-register-alloc-mode=pvc:-ze-opt-large-register-file"

// BAD_DEVICE: unsupported argument 'dg2:auto' to option '-ftarget-register-alloc-mode='
// BAD_MODE: unsupported argument 'pvc:superlarge' to option '-ftarget-register-alloc-mode='
// BAD_BOTH: unsupported argument 'dg2:superlarge' to option '-ftarget-register-alloc-mode='
