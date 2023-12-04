// Test -ftarget-compile-fast behaviors

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:    -fsycl-targets=spir64_gen -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_GEN %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl \
// RUN:     -fsycl-targets=spir64_gen -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_GEN %s

// TARGET_COMPILE_FAST_GEN: ocloc{{.*}} "-output"
// TARGET_COMPILE_FAST_GEN: "-options" "-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'"

// RUN: %clang -### -target x86_64-unknown-linux-gnu -fsycl \
// RUN:    -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_JIT %s
// RUN: %clang_cl -### --target=x86_64-pc-windows-msvc -fsycl \
// RUN:     -ftarget-compile-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefix=TARGET_COMPILE_FAST_JIT %s

// TARGET_COMPILE_FAST_JIT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-ftarget-compile-fast
