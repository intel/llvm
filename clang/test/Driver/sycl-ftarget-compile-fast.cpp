// Test -ftarget-compile-fast behaviors

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:    -fsycl-targets=intel_gpu_pvc -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_GEN %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:     -fsycl-targets=intel_gpu_pvc -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_GEN %s

// TARGET_COMPILE_FAST_GEN: llvm-offload-binary
// TARGET_COMPILE_FAST_GEN: compile-opts={{.*}}-options -igc_opts 'PartitionUnit=1,compile-opts=SubroutineThreshold=50000'

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:    -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_JIT %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:     -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_JIT %s

// TARGET_COMPILE_FAST_JIT: llvm-offload-binary
// TARGET_COMPILE_FAST_JIT: compile-opts={{.*}}-ftarget-compile-fast
