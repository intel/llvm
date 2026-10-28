// Tests that the SYCL toolchain links libclang_rt.builtins.bc for spirv64
// via -mlink-builtin-bitcode, that --no-offloadlib suppresses it, and that
// -nolibsycl does not suppress it (builtins are target infrastructure, not
// SYCL runtime).

// In intel/llvm, device libraries are only linked at compile time with the new
// offloading driver, and the default SYCL target is spir64 rather than
// spirv64, so pass --offload-new-driver and an explicit spirv64 target.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-BUILTIN-BC
// CHECK-BUILTIN-BC: "-triple" "spirv64-unknown-unknown"
// CHECK-BUILTIN-BC: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}spirv64-unknown-unknown{{[/\\]+}}libclang_rt.builtins.bc"

// A short --offload-targets= spelling finds the same per-target directory.
// The SYCL driver expands the spirv64 alias to spirv64-unknown-unknown.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver --offload-targets=spirv64 \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHORT-TRIPLE
// CHECK-SHORT-TRIPLE: "-triple" "spirv64-unknown-unknown"
// CHECK-SHORT-TRIPLE: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}spirv64-unknown-unknown{{[/\\]+}}libclang_rt.builtins.bc"

// The ITT annotation device libraries are not controlled by --no-offloadlib,
// so disable them with -fno-sycl-instrument-device-code.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --no-offloadlib \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-OFFLOADLIB
// CHECK-NO-OFFLOADLIB: "-cc1" "-triple" "spirv64-unknown-unknown"
// CHECK-NO-OFFLOADLIB-NOT: "-mlink-builtin-bitcode"

// -nolibsycl suppresses the SYCL runtime library but NOT compiler-rt builtins.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -nolibsycl \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOLIBSYCL
// CHECK-NOLIBSYCL: "-triple" "spirv64-unknown-unknown"
// CHECK-NOLIBSYCL: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}spirv64-unknown-unknown{{[/\\]+}}libclang_rt.builtins.bc"

// Test the flat layout used with LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF:
// lib/<os>/libclang_rt.builtins-spirv64.bc
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%S/Inputs/spirv64-sycl-flat-layout %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-FLAT-LAYOUT-BC
// CHECK-FLAT-LAYOUT-BC: "-triple" "spirv64-unknown-unknown"
// CHECK-FLAT-LAYOUT-BC: "-mlink-builtin-bitcode" "{{.*}}lib{{[/\\]+}}linux{{[/\\]+}}libclang_rt.builtins-spirv64.bc"

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%t/nonexistent %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MISSING
// CHECK-MISSING: error: no compiler-rt builtins bitcode library '{{.*}}libclang_rt.builtins{{.*}}.bc' found in the clang resource directory

// Compilations that stop at or before the compile phase never link bitcode, so
// a missing builtins library is not an error for them.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -E \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%t/nonexistent %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-CODEGEN
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl -fsyntax-only \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%t/nonexistent %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-CODEGEN
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fsycl --precompile \
// RUN:   --offload-new-driver --offload-targets=spirv64-unknown-unknown \
// RUN:   -resource-dir=%t/nonexistent -x c++-header %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-CODEGEN
// CHECK-NO-CODEGEN-NOT: error:
// CHECK-NO-CODEGEN-NOT: "-mlink-builtin-bitcode"
