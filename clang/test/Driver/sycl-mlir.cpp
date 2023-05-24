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
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-PHASES-LLVM-RAISE %s

// CHK-PHASES-LLVM-RAISE: 0: input, "{{.*}}.cpp", c++, (host-sycl)
// CHK-PHASES-LLVM-RAISE: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-LLVM-RAISE: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-LLVM-RAISE: 3: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-LLVM-RAISE: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LLVM-RAISE: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-LLVM-RAISE: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-syclmlir)" {5}, c++-cpp-output
// CHK-PHASES-LLVM-RAISE: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-LLVM-RAISE: 8: compiler, {7}, mlir, (host-sycl)
// CHK-PHASES-LLVM-RAISE: 9: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-LLVM-RAISE: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// CHK-PHASES-LLVM-RAISE: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {8}, "device-sycl (spir64-unknown-unknown-syclmlir)" {10}, mlir
// CHK-PHASES-LLVM-RAISE: 12: compiler, {11}, ir, (device-sycl)
// CHK-PHASES-LLVM-RAISE: 13: offload, "device-sycl (spir64-unknown-unknown-syclmlir)" {12}, ir

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
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-MLIR-RAISE %s
//
// CHK-PHASES-MLIR-RAISE: 0: input, "{{.*}}.cpp", c++, (host-sycl)
// CHK-PHASES-MLIR-RAISE: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-MLIR-RAISE: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-MLIR-RAISE: 3: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR-RAISE: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR-RAISE: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-MLIR-RAISE: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown-syclmlir)" {5}, c++-cpp-output
// CHK-PHASES-MLIR-RAISE: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-MLIR-RAISE: 8: compiler, {7}, mlir, (host-sycl)
// CHK-PHASES-MLIR-RAISE: 9: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR-RAISE: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR-RAISE: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {8}, "device-sycl (spir64-unknown-unknown-syclmlir)" {10}, mlir
// CHK-PHASES-MLIR-RAISE: 12: compiler, {11}, mlir, (device-sycl)
// CHK-PHASES-MLIR-RAISE: 13: offload, "device-sycl (spir64-unknown-unknown-syclmlir)" {12}, mlir

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-MLIR-NO-TRIPLE %s
//
// CHK-PHASES-MLIR-NO-TRIPLE: 0: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE: 2: compiler, {1}, mlir, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE: 3: offload, "device-sycl (spir64-unknown-unknown)" {2}, mlir

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-MLIR-NO-TRIPLE-RAISE %s

// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 0: input, "{{.*}}.cpp", c++, (host-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 3: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64-unknown-unknown)" {5}, c++-cpp-output
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 8: compiler, {7}, mlir, (host-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 9: input, "{{.*}}.cpp", c++, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 11: offload, "host-sycl (x86_64-unknown-linux-gnu)" {8}, "device-sycl (spir64-unknown-unknown)" {10}, mlir
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 12: compiler, {11}, mlir, (device-sycl)
// CHK-PHASES-MLIR-NO-TRIPLE-RAISE: 13: offload, "device-sycl (spir64-unknown-unknown)" {12}, mlir

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-LLVM %s
//
// CHK-BINDINGS-LLVM: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-LLVM-RAISE %s

// CHK-BINDINGS-LLVM-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"
// CHK-BINDINGS-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.bc"], output: "{{.*}}.bc"
// CHK-BINDINGS-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "mlir-translate", inputs: ["{{.*}}.bc"], output: "{{.*}}.mlir"
// CHK-BINDINGS-LLVM-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp", "{{.*}}.mlir"], output: "sycl-mlir-sycl-spir64-unknown-unknown-syclmlir.bc"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-MLIR %s
//
// CHK-BINDINGS-MLIR: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.mlir"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-MLIR-RAISE %s

// CHK-BINDINGS-MLIR-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"
// CHK-BINDINGS-MLIR-RAISE: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-MLIR-RAISE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.bc"], output: "{{.*}}.bc"
// CHK-BINDINGS-MLIR-RAISE: # "x86_64-unknown-linux-gnu" - "mlir-translate", inputs: ["{{.*}}.bc"], output: "{{.*}}.mlir"
// CHK-BINDINGS-MLIR-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp", "{{.*}}.mlir"], output: "sycl-mlir-sycl-spir64-unknown-unknown-syclmlir.mlir"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                 \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1     \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-FULL-LLVM %s
//
// CHK-BINDINGS-FULL-LLVM: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-LLVM: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-FULL-LLVM: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.bc"], output: "{{.*}}.o"
// CHK-BINDINGS-FULL-LLVM: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["{{.*}}.bc", "{{.*}}.o"], output: "{{.*}}.o"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                 \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                 \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1     \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-FULL-LLVM-RAISE %s
//
// CHK-BINDINGS-FULL-LLVM-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-FULL-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.bc"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "mlir-translate", inputs: ["{{.*}}.bc"], output: "{{.*}}.mlir"
// CHK-BINDINGS-FULL-LLVM-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp", "{{.*}}.mlir"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.bc"], output: "{{.*}}.o"
// CHK-BINDINGS-FULL-LLVM-RAISE: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["{{.*}}.bc", "{{.*}}.o"], output: "sycl-mlir.o"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-FULL-MLIR %s

// CHK-BINDINGS-FULL-MLIR: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.mlir"
// CHK-BINDINGS-FULL-MLIR: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-FULL-MLIR: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.mlir"], output: "{{.*}}.o"
// CHK-BINDINGS-FULL-MLIR: # "x86_64-unknown-linux-gnu" - "offload bundler", inputs: ["{{.*}}.mlir", "{{.*}}.o"], output: "sycl-mlir.o"

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL           \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-BINDINGS-FULL-MLIR-RAISE %s

// CHK-BINDINGS-FULL-MLIR-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-MLIR-RAISE: # "x86_64-unknown-linux-gnu" - "Append Footer to source", inputs: ["{{.*}}.cpp"], output: "{{.*}}.cpp"
// CHK-BINDINGS-FULL-MLIR-RAISE: # "x86_64-unknown-linux-gnu" - "clang", inputs: ["{{.*}}.cpp", "{{.*}}.bc"], output: "{{.*}}.bc"
// CHK-BINDINGS-FULL-MLIR-RAISE: # "x86_64-unknown-linux-gnu" - "mlir-translate", inputs: ["{{.*}}.bc"], output: "{{.*}}.mlir"
// CHK-BINDINGS-FULL-MLIR-RAISE: # "spir64-unknown-unknown-syclmlir" - "cgeist", inputs: ["{{.*}}.cpp", "{{.*}}.mlir"], output: "sycl-mlir-sycl-spir64-unknown-unknown-syclmlir.mlir"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-LLVM-BC %s
//
// CHK-INVOKE-LLVM-BC: "{{.*}}cgeist" "-emit-llvm" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-LLVM-BC-RAISE %s
//
// CHK-INVOKE-LLVM-BC-RAISE: "{{.*}}cgeist" "-emit-llvm" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"
// CHK-INVOKE-LLVM-BC-RAISE: "{{.*}}clang{{.*}}"
// CHK-INVOKE-LLVM-BC-RAISE: "{{.*}}mlir-translate" "-o" "{{.*}}.mlir" "--import-llvm" "{{.*}}.bc"
// CHK-INVOKE-LLVM-BC-RAISE: "{{.*}}cgeist" "-emit-llvm" "{{.*}}.cpp" "-sycl-use-host-module" "{{.*}}.mlir"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c  -S -emit-llvm       \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-LLVM %s
//
// CHK-INVOKE-LLVM: "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-o" "{{.*}}.ll" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c                      \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-LLVM-RAISE %s
//
// CHK-INVOKE-LLVM-RAISE: "{{.*}}cgeist" "-emit-llvm" "{{.*}}.cpp" "-o" "{{.*}}.bc"
// CHK-INVOKE-LLVM-RAISE: "{{.*}}clang{{.*}}"
// CHK-INVOKE-LLVM-RAISE: "{{.*}}mlir-translate" "-o" "{{.*}}.mlir" "--import-llvm" "{{.*}}.bc"
// CHK-INVOKE-LLVM-RAISE: "{{.*}}cgeist" "-emit-llvm" "{{.*}}.cpp" "-sycl-use-host-module" "{{.*}}.mlir" "-o" "{{.*}}.bc"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -emit-mlir          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1         \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-MLIR %s
//
// CHK-INVOKE-MLIR: "{{.*}}cgeist" "-S" "{{.*}}.cpp" "-o" "{{.*}}.mlir" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -emit-mlir          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s              \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                     \
// RUN: -o foo.mlir 2>&1                                               \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-MLIR-RAISE %s
//
// CHK-INVOKE-MLIR-RAISE: "{{.*}}cgeist" "-emit-llvm" "{{.*}}sycl-mlir.cpp" "-o" "{{.*}}.bc"
// CHK-INVOKE-MLIR-RAISE: "{{.*}}clang{{.*}}"
// CHK-INVOKE-MLIR-RAISE: "{{.*}}mlir-translate" "-o" "{{.*}}.mlir" "--import-llvm" "{{.*}}.bc"
// CHK-INVOKE-MLIR-RAISE: "{{.*}}cgeist" "-S" "{{.*}}.cpp" "-sycl-use-host-module" "{{.*}}.mlir" "-o" "{{.*}}.mlir"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -Xcgeist -S          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-ARG-PASS %s
//
// CHK-INVOKE-ARG-PASS: "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -Xcgeist -S          \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                                               \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-ARG-PASS-RAISE %s

// CHK-INVOKE-ARG-PASS-RAISE: "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"
// CHK-INVOKE-ARG-PASS-RAISE: "{{.*}}clang{{.*}}"
// CHK-INVOKE-ARG-PASS-RAISE: "{{.*}}mlir-translate" "-o" "{{.*}}.mlir" "--import-llvm" "{{.*}}.bc"
// CHK-INVOKE-ARG-PASS-RAISE:  "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-sycl-use-host-module" "{{.*}}.mlir"

// RUN: touch %t.o
// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL                          \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -Xcgeist -S             \
// RUN: -fsycl-raise-host -Xclang -opaque-pointers                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %t.o %s 2>&1     \
// RUN: | FileCheck -check-prefix=CHK-UNBUNDLER-RAISE %s

// CHK-UNBUNDLER-RAISE: "{{.*}}clang-offload-bundler" "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown-syclmlir"

// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only       \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -Xcgeist -S          \
// RUN: -fsycl-raise-host -Xclang -no-opaque-pointers                   \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-RAISE-TYPED-PTR %s

// CHK-RAISE-TYPED-PTR: error: invalid argument '-fsycl-raise-host' only allowed with '-Xclang -opaque-pointers'
