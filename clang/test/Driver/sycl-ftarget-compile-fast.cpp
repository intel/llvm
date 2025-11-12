// Test -ftarget-compile-fast behaviors

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_GEN %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:     -fsycl-targets=spir64_gen -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_GEN %s

<<<<<<< HEAD

// Due to how clang-offload-packager works, if we want a value in a key-value pair to
// have a comma, we need to specify the key twice, once for each part of the value
// separated by the comma. clang-offload-packager will then combine the two parts into a single
// value with a comma in between.
// TARGET_COMPILE_FAST_GEN: clang-offload-packager
// TARGET_COMPILE_FAST_GEN: compile-opts={{.*}}-options -igc_opts 'PartitionUnit=1
// TARGET_COMPILE_FAST_GEN-SAME: compile-opts=SubroutineThreshold=50000'
=======
// TARGET_COMPILE_FAST_GEN: llvm-offload-binary
// TARGET_COMPILE_FAST_GEN: compile-opts={{.*}}-options -igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'
>>>>>>> intel/sycl

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:    -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_JIT %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:     -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_JIT %s

// TARGET_COMPILE_FAST_JIT: llvm-offload-binary
// TARGET_COMPILE_FAST_JIT: compile-opts={{.*}}-ftarget-compile-fast
