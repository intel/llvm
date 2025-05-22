/// Tests behaviors of -fsyclbin

/// -fsyclbin is only used with the new offloading model.
// RUN: %clangxx -fsycl -fsyclbin --no-offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=UNUSED
// UNUSED: warning: argument unused during compilation: '-fsyclbin'

/// -fsyclbin -fsycl-device-only usage.  -fsycl-device-only will 'win' and
/// -fsyclbin is effectively ignored.
// RUN: %clangxx -fsycl-device-only -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=SYCLBIN_UNUSED
// RUN: %clang_cl -fsycl-device-only -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=SYCLBIN_UNUSED
// SYCLBIN_UNUSED: warning: argument unused during compilation: '-fsyclbin'

/// Check tool invocation contents.
// RUN: %clangxx -fsycl -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_DEFAULT
// RUN: %clangxx -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_DEFAULT
// RUN: %clang_cl -fsycl -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_DEFAULT
// CHECK_TOOLS_DEFAULT: clang-offload-packager
// CHECK_TOOLS_DEFAULT-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_DEFAULT-SAME: kind=sycl
// CHECK_TOOLS_DEFAULT: clang-linker-wrapper
// CHECK_TOOLS_DEFAULT-SAME: --syclbin=executable

/// Check tool invocation contents.
// RUN: %clangxx -fsycl -fsyclbin=input --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_INPUT
// RUN: %clangxx -fsyclbin=input --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_INPUT
// RUN: %clang_cl -fsycl -fsyclbin=input --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_INPUT
// CHECK_TOOLS_INPUT: clang-offload-packager
// CHECK_TOOLS_INPUT-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_INPUT-SAME: kind=sycl
// CHECK_TOOLS_INPUT: clang-linker-wrapper
// CHECK_TOOLS_INPUT-SAME: --syclbin=input

/// Check tool invocation contents.
// RUN: %clangxx -fsycl -fsyclbin=object --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_OBJECT
// RUN: %clangxx -fsyclbin=object --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_OBJECT
// RUN: %clang_cl -fsycl -fsyclbin=object --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_OBJECT
// CHECK_TOOLS_OBJECT: clang-offload-packager
// CHECK_TOOLS_OBJECT-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_OBJECT-SAME: kind=sycl
// CHECK_TOOLS_OBJECT: clang-linker-wrapper
// CHECK_TOOLS_OBJECT-SAME: --syclbin=object

/// Check tool invocation contents.
// RUN: %clangxx -fsycl -fsyclbin=executable --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clangxx -fsyclbin=executable --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// RUN: %clang_cl -fsycl -fsyclbin=executable --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_EXECUTABLE
// CHECK_TOOLS_EXECUTABLE: clang-offload-packager
// CHECK_TOOLS_EXECUTABLE-SAME: --image=file={{.*}}.bc,triple=spir64-unknown-unknown
// CHECK_TOOLS_EXECUTABLE-SAME: kind=sycl
// CHECK_TOOLS_EXECUTABLE: clang-linker-wrapper
// CHECK_TOOLS_EXECUTABLE-SAME: --syclbin=executable

/// Check compilation phases, only device compile should be performed
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl -fsyclbin \
// RUN:   --offload-new-driver %s -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsyclbin \
// RUN:   --offload-new-driver %s -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// CHECK_PHASES: 0: input, "{{.*}}", c++, (device-sycl)
// CHECK_PHASES: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// CHECK_PHASES: 2: compiler, {1}, ir, (device-sycl)
// CHECK_PHASES: 3: backend, {2}, ir, (device-sycl)
// CHECK_PHASES: 4: offload, "device-sycl (spir64-unknown-unknown)" {3}, ir
// CHECK_PHASES: 5: clang-offload-packager, {4}, image, (device-sycl)
// CHECK_PHASES: 6: clang-linker-wrapper, {5}, image, (device-sycl)
// CHECK_PHASES: 7: offload, "device-sycl (x86_64-unknown-linux-gnu)" {6}, none

/// Check the output file names (file.syclbin, or -o <file>)
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsyclbin \
// RUN:   --offload-new-driver -o file.syclbin %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NAMED_OUTPUT
// CHECK_NAMED_OUTPUT: clang-linker-wrapper
// CHECK_NAMED_OUTPUT-SAME: "-o" "file.syclbin"

// RUN: %clang_cl -fsyclbin --offload-new-driver -o file.syclbin %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NAMED_OUTPUT_WIN
// CHECK_NAMED_OUTPUT_WIN: clang-linker-wrapper
// CHECK_NAMED_OUTPUT_WIN-SAME: "-out:file.syclbin"

/// For Linux - the default is 'a.out' so the syclbin file is 'a.syclbin'
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsyclbin \
// RUN:   --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_LINUX_DEFAULT_OUTPUT
// CHECK_LINUX_DEFAULT_OUTPUT: clang-linker-wrapper
// CHECK_LINUX_DEFAULT_OUTPUT-SAME: "-o" "a.syclbin"

/// For Windows - the default is based on the source file so the syclbin file
/// is 'fsyclbin.syclbin'
// RUN: %clang_cl -fsyclbin --offload-new-driver %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_WIN_DEFAULT_OUTPUT
// CHECK_WIN_DEFAULT_OUTPUT: clang-linker-wrapper
// CHECK_WIN_DEFAULT_OUTPUT-SAME: "-out:{{.*}}fsyclbin.syclbin"
