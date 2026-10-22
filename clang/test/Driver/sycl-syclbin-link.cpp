/// Tests behaviors of -fsycl-link with SYCLBIN inputs.

// RUN: rm -rf %t && mkdir -p %t
// RUN: touch %t/a.syclbin %t/b.syclbin %t/c.o

/// Linking SYCLBIN files produces an executable-state SYCLBIN. The wrapper is
/// told to link for the default JIT target and no device link is performed.
/// -fsycl and --offload-new-driver are implied by linking SYCLBIN files.
// RUN: %clangxx -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_JIT \
// RUN:   --implicit-check-not=--sycl-device-link
// CHECK_JIT: clang-linker-wrapper
// CHECK_JIT-SAME: "--syclbin=executable"
// CHECK_JIT-SAME: "--syclbin-link-target=spir64-unknown-unknown"
/// For Linux - the default output name is 'a.syclbin'.
// CHECK_JIT-SAME: "-o" "a.syclbin"
// CHECK_JIT-SAME: "{{.*}}a.syclbin" "{{.*}}b.syclbin"

/// The output file can be named with -o.
// RUN: %clangxx -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -o linked.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NAMED_OUTPUT
// CHECK_NAMED_OUTPUT: clang-linker-wrapper
// CHECK_NAMED_OUTPUT-SAME: "--syclbin=executable"
// CHECK_NAMED_OUTPUT-SAME: "-o" "linked.syclbin"

// RUN: %clang_cl -fsycl-link \
// RUN:   %t/a.syclbin %t/b.syclbin -o linked.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NAMED_OUTPUT_WIN
// CHECK_NAMED_OUTPUT_WIN: clang-linker-wrapper
// CHECK_NAMED_OUTPUT_WIN-SAME: "--syclbin=executable"
// CHECK_NAMED_OUTPUT_WIN-SAME: "-out:linked.syclbin"

// RUN: %clang_cl -fsycl-link \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_WIN_DEFAULT_OUTPUT
// CHECK_WIN_DEFAULT_OUTPUT: clang-linker-wrapper
// CHECK_WIN_DEFAULT_OUTPUT-SAME: "-out:a.syclbin"

/// An AOT target can be selected with --offload-arch.
// RUN: %clangxx -fsycl-link --offload-arch=pvc \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_AOT_GPU
// CHECK_AOT_GPU: clang-linker-wrapper
// CHECK_AOT_GPU-SAME: "--syclbin=executable"
// CHECK_AOT_GPU-SAME: "--syclbin-link-target=spir64_gen-unknown-unknown=pvc"

// RUN: %clangxx -fsycl-link --offload-arch=corei7 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_AOT_CPU
// CHECK_AOT_CPU: clang-linker-wrapper
// CHECK_AOT_CPU-SAME: "--syclbin=executable"
// CHECK_AOT_CPU-SAME: "--syclbin-link-target=spir64_x86_64-unknown-unknown=corei7"

/// Multiple architectures result in one link target each.
// RUN: %clangxx -fsycl-link \
// RUN:   --offload-arch=pvc,bmg_g21,corei7 \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_AOT_MULTI
// CHECK_AOT_MULTI: clang-linker-wrapper
// CHECK_AOT_MULTI-DAG: "--syclbin-link-target=spir64_gen-unknown-unknown=pvc"
// CHECK_AOT_MULTI-DAG: "--syclbin-link-target=spir64_gen-unknown-unknown=bmg_g21"
// CHECK_AOT_MULTI-DAG: "--syclbin-link-target=spir64_x86_64-unknown-unknown=corei7"

/// Only the link step is performed.
// RUN: %clangxx -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -ccc-print-phases 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASES
// CHECK_PHASES: 0: input, "{{.*}}a.syclbin", object, (host-sycl)
// CHECK_PHASES: 1: input, "{{.*}}b.syclbin", object, (host-sycl)
// CHECK_PHASES: 2: clang-linker-wrapper, {0, 1}, image, (host-sycl)
// CHECK_PHASES-NOT: {{[0-9]+}}:

/// Phase-limiting options stop before the link step, so the SYCLBIN files are
/// unused, as object files would be, and no link is performed.
// RUN: %clangxx -fsycl-link -c \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASE_LIMIT \
// RUN:   --implicit-check-not=clang-linker-wrapper
// RUN: %clangxx -fsycl-link -S \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASE_LIMIT \
// RUN:   --implicit-check-not=clang-linker-wrapper
// RUN: %clangxx -fsycl-link -E \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASE_LIMIT \
// RUN:   --implicit-check-not=clang-linker-wrapper
// RUN: %clangxx -fsycl-link -fsyntax-only \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_PHASE_LIMIT \
// RUN:   --implicit-check-not=clang-linker-wrapper
// CHECK_PHASE_LIMIT: warning: {{.*}}a.syclbin: 'linker' input unused
// CHECK_PHASE_LIMIT: warning: {{.*}}b.syclbin: 'linker' input unused

/// SYCLBIN files cannot be linked together with other inputs.
// RUN: not %clangxx -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/c.o -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_MIXED
// CHECK_MIXED: error: SYCLBIN file '{{.*}}a.syclbin' cannot be linked with non-SYCLBIN input '{{.*}}c.o' when using '-fsycl-link'

/// Explicitly passing -fsycl and --offload-new-driver is also accepted.
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_JIT \
// RUN:   --implicit-check-not=--sycl-device-link

/// Linking SYCLBIN files requires the new offloading model.
// RUN: not %clangxx --no-offload-new-driver -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_OLD_DRIVER
// CHECK_OLD_DRIVER: error: linking SYCLBIN files with '-fsycl-link' requires '--offload-new-driver'

/// Linking SYCLBIN files requires SYCL offloading.
// RUN: not %clangxx -fno-sycl -fsycl-link \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_SYCL
// CHECK_NO_SYCL: error: '-fsycl-link' must be used in conjunction with '-fsycl' to enable offloading

/// Without -fsycl-link, SYCLBIN inputs are not handled specially.
// RUN: %clangxx -fsycl --offload-new-driver \
// RUN:   --target=x86_64-unknown-linux-gnu \
// RUN:   %t/a.syclbin %t/b.syclbin -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_NO_LINK
// CHECK_NO_LINK: clang-linker-wrapper
// CHECK_NO_LINK-NOT: --syclbin-link-target
