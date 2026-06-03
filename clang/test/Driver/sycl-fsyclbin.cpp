/// Tests behaviors of -fsyclbin

/// -fsyclbin is only used with the new offloading model.
// RUN: %clangxx -fsycl -fsyclbin --no-offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning: argument unused during compilation: '-fsyclbin'

/// -fsyclbin -fsycl-device-only usage.  -fsycl-device-only will 'win' and
/// -fsyclbin is effectively ignored.
// RUN: %clangxx -fsycl-device-only -fsyclbin --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=SYCLBIN_UNUSED
// RUN: %clang_cl -fsycl-device-only -fsyclbin --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=SYCLBIN_UNUSED
// SYCLBIN_UNUSED: warning: argument unused during compilation: '-fsyclbin'

/// Check tool invocation contents.
// RUN: %clangxx -fsycl -fsyclbin=input --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_INPUT
// RUN: %clangxx -fsyclbin=input --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_INPUT
// RUN: %clang_cl -fsycl -fsyclbin=input --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_INPUT
// CHECK_TOOLS_INPUT: llvm-offload-binary
// CHECK_TOOLS_INPUT-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_INPUT-SAME: kind=sycl
// CHECK_TOOLS_INPUT: clang-linker-wrapper
// CHECK_TOOLS_INPUT-SAME: --syclbin=input

// RUN: %clangxx -fsycl -fsyclbin=object --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_OBJECT
// RUN: %clangxx -fsyclbin=object --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_OBJECT
// RUN: %clang_cl -fsycl -fsyclbin=object --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_OBJECT
// CHECK_TOOLS_OBJECT: llvm-offload-binary
// CHECK_TOOLS_OBJECT-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_OBJECT-SAME: kind=sycl
// CHECK_TOOLS_OBJECT: clang-linker-wrapper
// CHECK_TOOLS_OBJECT-SAME: --syclbin=object

// RUN: %clangxx -fsycl -fsyclbin=executable --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clangxx -fsyclbin=executable --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clang_cl -fsycl -fsyclbin=executable --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clangxx -fsycl -fsyclbin --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clangxx -fsyclbin --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clang_cl -fsycl -fsyclbin --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// CHECK_TOOLS_EXECUTABLE: llvm-offload-binary
// CHECK_TOOLS_EXECUTABLE-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_EXECUTABLE-SAME: kind=sycl
// CHECK_TOOLS_EXECUTABLE: clang-linker-wrapper
// CHECK_TOOLS_EXECUTABLE-SAME: --syclbin=executable

/// Check compilation phases, only device compile should be performed
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fsyclbin \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsyclbin \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// CHECK_PHASES: 0: input, "{{.*}}", c++, (device-sycl)
// CHECK_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHECK_PHASES: 2: compiler, {1}, ir, (device-sycl)
// CHECK_PHASES: 3: backend, {2}, ir, (device-sycl)
// CHECK_PHASES: 4: offload, "device-sycl (spir64-unknown-unknown)" {3}, ir
// CHECK_PHASES: 5: llvm-offload-binary, {4}, image, (device-sycl)
// CHECK_PHASES: 6: clang-linker-wrapper, {5}, image, (device-sycl)
// CHECK_PHASES: 7: offload, "device-sycl (x86_64-unknown-linux-gnu)" {6}, none

/// Check the output file names (file.syclbin, or -o <file>)
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsyclbin \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL -o file.syclbin %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NAMED_OUTPUT
// CHECK_NAMED_OUTPUT: clang-linker-wrapper
// CHECK_NAMED_OUTPUT-SAME: "-o" "file.syclbin"

// RUN: %clang_cl -fsyclbin --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL -o file.syclbin %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NAMED_OUTPUT_WIN
// CHECK_NAMED_OUTPUT_WIN: clang-linker-wrapper
// CHECK_NAMED_OUTPUT_WIN-SAME: "-out:file.syclbin"

/// For Linux - the default is 'a.out' so the syclbin file is 'a.syclbin'
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsyclbin \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_LINUX_DEFAULT_OUTPUT
// CHECK_LINUX_DEFAULT_OUTPUT: clang-linker-wrapper
// CHECK_LINUX_DEFAULT_OUTPUT-SAME: "-o" "a.syclbin"

/// For Windows - the default is based on the source file so the syclbin file
/// is 'fsyclbin.syclbin'
// RUN: %clang_cl -fsyclbin --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_WIN_DEFAULT_OUTPUT
// CHECK_WIN_DEFAULT_OUTPUT: clang-linker-wrapper
// CHECK_WIN_DEFAULT_OUTPUT-SAME: "-out:{{.*}}fsyclbin.syclbin"

/// -fsyclbin=input and -fsyclbin=object imply
/// -fsycl-allow-device-image-dependencies, since cross-image symbol
/// resolution is required for those bundle states.
// RUN: %clangxx -fsyclbin=input --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEPS_IMPLIED
// RUN: %clangxx -fsyclbin=object --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEPS_IMPLIED
// CHECK_DEPS_IMPLIED: clang-linker-wrapper
// CHECK_DEPS_IMPLIED-SAME: "-sycl-allow-device-image-dependencies"
// CHECK_DEPS_IMPLIED-SAME: --sycl-post-link-options={{.*}}-allow-device-image-dependencies

/// -fsyclbin=executable does not imply
/// -fsycl-allow-device-image-dependencies, since the resulting bundle
/// is fully linked and does not participate in cross-image link.
// RUN: %clangxx -fsyclbin=executable --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEPS_NOT_IMPLIED
// CHECK_DEPS_NOT_IMPLIED-NOT: -sycl-allow-device-image-dependencies

/// For AOT spir64_gen, -fsyclbin=input/object additionally implies
/// -library-compilation in the backend options so symbols stay
/// externally visible for runtime cross-image resolution.
// RUN: %clangxx -fsyclbin=object -fsycl-targets=spir64_gen \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_LIBCOMP_IMPLIED
// CHECK_LIBCOMP_IMPLIED: -library-compilation

// RUN: %clangxx -fsyclbin=executable -fsycl-targets=spir64_gen \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_LIBCOMP_NOT_IMPLIED
// CHECK_LIBCOMP_NOT_IMPLIED-NOT: -library-compilation

/// Combining -fsyclbin=input/object with
/// -fno-sycl-allow-device-image-dependencies is a contradiction: an
/// object SYCLBIN that forbids cross-image dependencies has no use
/// case, so the driver emits an error and redirects users to
/// -fsyclbin=executable.
// RUN: not %clangxx -fsyclbin=object -fno-sycl-allow-device-image-dependencies \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEPS_CONFLICT
// RUN: not %clangxx -fsyclbin=input -fno-sycl-allow-device-image-dependencies \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEPS_CONFLICT
// CHECK_DEPS_CONFLICT: error: invalid argument '-fno-sycl-allow-device-image-dependencies' not allowed with '-fsyclbin={{input|object}}'

/// Combining -fsyclbin=executable with
/// -fsycl-allow-device-image-dependencies is harmless but pointless,
/// so the driver emits a warning.
// RUN: %clangxx -fsyclbin=executable -fsycl-allow-device-image-dependencies \
// RUN:   --offload-new-driver --sysroot=%S/Inputs/SYCL %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEPS_REDUNDANT
// CHECK_DEPS_REDUNDANT: warning: invalid argument '-fsycl-allow-device-image-dependencies' not allowed with '-fsyclbin=executable'
