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
// RUN: -fsycl-raise-host                                               \
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

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL               \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only     \
// RUN: -fsycl-raise-host                                                 \
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
// RUN: -fsycl-raise-host                                               \
// RUN: -fsycl-targets=spir64-unknown-unknown -emit-mlir %s 2>&1        \
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
// RUN: -fsycl-raise-host                                               \
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

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-device-only     \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir %s 2>&1 \
// RUN: -fsycl-raise-host                                                 \
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
// RUN: -fsycl-raise-host                                          \
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

// RUN: %clangxx -ccc-print-bindings --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-raise-host      \
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
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-raise-host    \
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
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -fsycl-raise-host    \
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
// RUN: -fsycl-raise-host -o foo.mlir 2>&1                             \
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
// RUN: -fsycl-raise-host                                               \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-INVOKE-ARG-PASS-RAISE %s

// CHK-INVOKE-ARG-PASS-RAISE: "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-o" "{{.*}}.bc" "--args" "-cc1"
// CHK-INVOKE-ARG-PASS-RAISE: "{{.*}}clang{{.*}}"
// CHK-INVOKE-ARG-PASS-RAISE: "{{.*}}mlir-translate" "-o" "{{.*}}.mlir" "--import-llvm" "{{.*}}.bc"
// CHK-INVOKE-ARG-PASS-RAISE:  "{{.*}}cgeist" "-emit-llvm" "-S" "{{.*}}.cpp" "-sycl-use-host-module" "{{.*}}.mlir"

// RUN: touch %t.o
// RUN: %clangxx -### --sysroot=%S/Inputs/SYCL                          \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -Xcgeist -S             \
// RUN: -fsycl-raise-host                                               \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %t.o %s 2>&1     \
// RUN: | FileCheck -check-prefix=CHK-UNBUNDLER-RAISE %s

// CHK-UNBUNDLER-RAISE: "{{.*}}clang-offload-bundler" "-type=o" "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown"

// RUN: not %clangxx -### --sysroot=%S/Inputs/SYCL -fsycl-device-only   \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -c -Xcgeist -S          \
// RUN: -fsycl-raise-host -fsycl-host-compiler=g++                      \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN: | FileCheck -check-prefix=CHK-RAISE-HOST-CCMP %s

// CHK-RAISE-HOST-CCMP: error: the combination of '-fsycl-raise-host' and '-fsycl-host-compiler=' is incompatible

// Make sure -fsycl-raise-host is ignored with -fsycl-link and -fsycl-link-targets

// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl                         \
// RUN: -fsycl-raise-host -fsycl-link -fno-sycl-device-lib=all          \
// RUN: -fsycl-targets=spir64-unknown-unknown-syclmlir %s 2>&1          \
// RUN:  | FileCheck -check-prefix=CHK-SYCL-LINK -check-prefix=CHK-SYCL-LINK-MLIR %s
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL             \
// RUN: -target x86_64-unknown-linux-gnu -fsycl                         \
// RUN: -fsycl-link  -fno-sycl-device-lib=all                           \
// RUN: -fsycl-targets=spir64-unknown-unknown %s  2>&1                  \
// RUN:  | FileCheck -check-prefix=CHK-SYCL-LINK -check-prefix=CHK-SYCL-LINK-LLVM %s

// CHK-SYCL-LINK: 0: input, "{{.*}}sycl-mlir.cpp", c++, (device-sycl)
// CHK-SYCL-LINK: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-SYCL-LINK: 2: compiler, {1}, ir, (device-sycl)
// CHK-SYCL-LINK: 3: linker, {2}, image, (device-sycl)
// CHK-SYCL-LINK: 4: input, "{{.*}}libsycl-itt-user-wrappers.o", object
// CHK-SYCL-LINK: 5: clang-offload-unbundler, {4}, object
// CHK-SYCL-LINK-MLIR: 6: offload, " (spir64-unknown-unknown-syclmlir)" {5}, object
// CHK-SYCL-LINK-LLVM: 6: offload, " (spir64-unknown-unknown)" {5}, object
// CHK-SYCL-LINK: 7: input, "{{.*}}libsycl-itt-compiler-wrappers.o", object
// CHK-SYCL-LINK: 8: clang-offload-unbundler, {7}, object
// CHK-SYCL-LINK-MLIR: 9: offload, " (spir64-unknown-unknown-syclmlir)" {8}, object
// CHK-SYCL-LINK-LLVM: 9: offload, " (spir64-unknown-unknown)" {8}, object
// CHK-SYCL-LINK: 10: input, "{{.*}}libsycl-itt-stubs.o", object
// CHK-SYCL-LINK: 11: clang-offload-unbundler, {10}, object
// CHK-SYCL-LINK-MLIR: 12: offload, " (spir64-unknown-unknown-syclmlir)" {11}, object
// CHK-SYCL-LINK-LLVM: 12: offload, " (spir64-unknown-unknown)" {11}, object
// CHK-SYCL-LINK: 13: linker, {3, 6, 9, 12}, ir, (device-sycl)
// CHK-SYCL-LINK: 14: sycl-post-link, {13}, ir, (device-sycl)
// CHK-SYCL-LINK: 15: file-table-tform, {14}, tempfilelist, (device-sycl)
// CHK-SYCL-LINK: 16: llvm-spirv, {15}, tempfilelist, (device-sycl)
// CHK-SYCL-LINK: 17: file-table-tform, {14, 16}, tempfiletable, (device-sycl)
// CHK-SYCL-LINK: 18: clang-offload-wrapper, {17}, object, (device-sycl)
// CHK-SYCL-LINK-MLIR: 19: offload, "device-sycl (spir64-unknown-unknown-syclmlir)" {18}, object
// CHK-SYCL-LINK-LLVM: 19: offload, "device-sycl (spir64-unknown-unknown)" {18}, object

// RUN: %clangxx -ccc-print-phases                                      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl                         \
// RUN: -fsycl-raise-host                                               \
// RUN: -fsycl-link-targets=spir64-unknown-unknown-syclmlir %s 2>&1     \
// RUN:  | FileCheck -check-prefix=CHK-SYCL-LINK-TARGET -check-prefix=CHK-SYCL-LINK-TARGET-MLIR %s
// RUN: %clangxx -ccc-print-phases                                      \
// RUN: -target x86_64-unknown-linux-gnu -fsycl                         \
// RUN: -fsycl-link-targets=spir64-unknown-unknown %s  2>&1             \
// RUN:  | FileCheck -check-prefix=CHK-SYCL-LINK-TARGET -check-prefix=CHK-SYCL-LINK-TARGET-LLVM %s

// CHK-SYCL-LINK-TARGET: 0: input, "{{.*}}sycl-mlir.cpp", c++, (device-sycl)
// CHK-SYCL-LINK-TARGET: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHK-SYCL-LINK-TARGET: 2: compiler, {1}, ir, (device-sycl)
// CHK-SYCL-LINK-TARGET: 3: linker, {2}, image, (device-sycl)
// CHK-SYCL-LINK-TARGET: 4: llvm-spirv, {3}, image, (device-sycl)
// CHK-SYCL-LINK-TARGET-MLIR: 5: offload, "device-sycl (spir64-unknown-unknown-syclmlir)" {4}, image
// CHK-SYCL-LINK-TARGET-LLVM: 5: offload, "device-sycl (spir64-unknown-unknown)" {4}, image
