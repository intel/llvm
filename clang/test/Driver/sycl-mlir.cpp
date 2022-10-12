/// Tests specific to -syclmlir

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-PHASES-LLVM %s
//
// CHK-PHASES-LLVM: 0: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-LLVM: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LLVM: 2: compiler, {1}, ir, (device-sycl)
// CHK-PHASES-LLVM: 3: offload, "device-sycl (spir64-unknown-unknown-syclmlir)" {2}, ir

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-MLIR %s
//
// CHK-PHASES-MLIR: 0: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR: 2: compiler, {1}, mlir, (device-sycl)
// CHK-PHASES-MLIR: 3: offload, "device-sycl (spir64-unknown-unknown-syclmlir)" {2}, mlir


// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-MLIR-NO-TRIPLE %s
//
// CHK-PHASES-MLIR-NO-TRIPLE: 0: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE: 2: compiler, {1}, mlir, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, mlir

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-LLVM %s
//
// CHK-BINDINGS-LLVM: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-MLIR %s
//
// CHK-BINDINGS-MLIR: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.mlir"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                 \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1     \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-FULL-LLVM %s
//
// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-FULL-LLVM %s
//
// CHK-BINDINGS-FULL-LLVM: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-LLVM: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-FULL-LLVM: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.bc"], output: "{{.*}}.o"
// CHK-BINDINGS-FULL-LLVM: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["{{.*}}.bc", "{{.*}}.o"], output: "{{.*}}.o"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-LLVM-BC %s
//
// CHK-INVOKE-LLVM-BC: "{{.*}}cgeist" "-emit-llvm" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c  -S -emit-llvm       \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-LLVM %s
//
// CHK-INVOKE-LLVM: "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-o" "{{.*}}.ll" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -emit-mlir          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1         \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-MLIR %s
//
// CHK-INVOKE-MLIR: "{{.*}}cgeist" "-S" "{{.*}}.cpp" "-o" "{{.*}}.mlir" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -emit-mlir          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s              \
// RUN: -o foo.mlir 2>&1                                               \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-MLIR %s
//
// CHK-INVOKE-MLIR-O: "{{.*}}cgeist" "-S" "{{.*}}.cpp" "-o" "foo.mlir" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -Xcgeist -S          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-ARG-PASS %s
//
// CHK-INVOKE-ARG-PASS: "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"
