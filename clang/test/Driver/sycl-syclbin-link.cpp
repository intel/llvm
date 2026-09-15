/// Tests linking SYCLBIN files into a single SYCLBIN file in executable state
/// with -fsycl-link.

/// The clang-linker-wrapper performs the link and is told to emit a SYCLBIN
/// file in executable state, as well as the target to compile the device code
/// for.  '-fsycl' and '--offload-new-driver' do not have to be passed, as a
/// SYCLBIN input implies both.  '--sycl-device-link' is not passed either, as
/// the output is a SYCLBIN container rather than a plain device image.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS \
// RUN:   --implicit-check-not=--sycl-device-link
// CHECK_TOOLS: clang-linker-wrapper
// CHECK_TOOLS-SAME: "--syclbin=executable"
// CHECK_TOOLS-SAME: "--syclbin-link-target=spir64_gen-unknown-unknown=pvc"
// CHECK_TOOLS-SAME: "-o" "out.syclbin"

/// A single SYCLBIN input is a valid link too.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS

/// The device code is compiled ahead of time for every requested architecture.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch=pvc,bmg_g21 --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_MULTI_ARCH
// CHECK_MULTI_ARCH: clang-linker-wrapper
// CHECK_MULTI_ARCH-SAME: "--syclbin-link-target=spir64_gen-unknown-unknown=bmg_g21"
// CHECK_MULTI_ARCH-SAME: "--syclbin-link-target=spir64_gen-unknown-unknown=pvc"

// RUN: %clang_cl -fsycl-link --offload-arch=pvc \
// RUN:   /clang:--sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_TOOLS_WIN
// CHECK_TOOLS_WIN: clang-linker-wrapper
// CHECK_TOOLS_WIN-SAME: "--syclbin=executable"
// CHECK_TOOLS_WIN-SAME: "--syclbin-link-target=spir64_gen-unknown-unknown=pvc"
// CHECK_TOOLS_WIN-SAME: "-out:out.syclbin"

/// The SYCLBIN files are pure device code containers, so no host compilation or
/// host link takes place.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin \
// RUN:   -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// CHECK_PHASES: 0: input, "{{.*}}a.syclbin", syclbin, (host-sycl)
// CHECK_PHASES: 1: input, "{{.*}}b.syclbin", syclbin, (host-sycl)
// CHECK_PHASES: 2: clang-linker-wrapper, {0, 1}, image, (host-sycl)

/// The device code in a SYCLBIN file is not tied to a device, so the
/// architectures to compile it for have to be named explicitly.
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_OFFLOAD_ARCH
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch= --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_OFFLOAD_ARCH
// CHECK_NO_OFFLOAD_ARCH: error: SYCLBIN input file '{{.*}}a.syclbin' requires '--offload-arch'

/// Without '-o' the output file name is derived the same way as with -fsyclbin,
/// i.e. 'a.syclbin' on Linux and '<basename>.syclbin' in CL mode.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEFAULT_OUTPUT
// CHECK_DEFAULT_OUTPUT: clang-linker-wrapper
// CHECK_DEFAULT_OUTPUT-SAME: "-o" "a.syclbin"

// RUN: %clang_cl -fsycl-link --offload-arch=pvc \
// RUN:   /clang:--sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/b.syclbin %S/Inputs/SYCL/a.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_DEFAULT_OUTPUT_WIN
// CHECK_DEFAULT_OUTPUT_WIN: clang-linker-wrapper
// CHECK_DEFAULT_OUTPUT_WIN-SAME: "-out:b.syclbin"

/// Linking SYCLBIN files requires a device-only link.
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_SYCL_LINK
// CHECK_NO_SYCL_LINK: error: SYCLBIN input file '{{.*}}a.syclbin' requires '-fsycl-link'

/// The SYCL offloading toolchain and the new offloading driver are implied by a
/// SYCLBIN input, so explicitly turning either of them off is an error.
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -fno-sycl -fsycl-link \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_FSYCL
// CHECK_NO_FSYCL: error: SYCLBIN input file '{{.*}}a.syclbin' requires '-fsycl'

// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --no-offload-new-driver --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/b.syclbin -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_NEW_DRIVER
// CHECK_NO_NEW_DRIVER: error: SYCLBIN input file '{{.*}}a.syclbin' requires '--offload-new-driver'

/// SYCLBIN files contain no host code, so they cannot participate in a link
/// with other kinds of inputs.
// RUN: not %clangxx --target=x86_64-unknown-linux-gnu -fsycl-link \
// RUN:   --offload-arch=pvc --sysroot=%S/Inputs/SYCL \
// RUN:   %S/Inputs/SYCL/a.syclbin %S/Inputs/SYCL/objlin64.o -o out.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_MIXED_INPUTS
// CHECK_MIXED_INPUTS: error: SYCLBIN input file '{{.*}}a.syclbin' cannot be linked with non-SYCLBIN input file '{{.*}}objlin64.o'
