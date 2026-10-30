/// Check that SYCL device code is extracted from static archives when no other
/// input provides device code, such as when all objects are host only.
// REQUIRES: system-linux, x86-registered-target
// REQUIRES: spirv-to-ir-wrapper, sycl-post-link

// RUN: %clang %s -fsycl -fsycl-targets=spir64-unknown-unknown -c --offload-new-driver --no-offloadlib -fno-sycl-instrument-device-code -o %t.o
// RUN: %clang %s -fsycl -fsycl-targets=spir64_x86_64 -c --offload-new-driver --no-offloadlib -fno-sycl-instrument-device-code -o %t_cpu.o
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t_host.o
// RUN: rm -f %t.a %t_cpu.a && llvm-ar rcs %t.a %t.o && llvm-ar rcs %t_cpu.a %t_cpu.o

/// Host only objects with an archive: the archive device code is extracted.
// RUN: clang-linker-wrapper --dry-run --linker-path=/usr/bin/ld -o /dev/null %t_host.o %t.a 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EXTRACT %s
// CHK-EXTRACT: sycl-post-link
// CHK-EXTRACT: offload-wrapper: output:

/// Objects providing device code for another target: the archive device code
/// is not extracted.
// RUN: clang-linker-wrapper --dry-run --linker-path=/usr/bin/ld -o /dev/null %t_cpu.o %t.a 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-CPU-ONLY %s
// CHK-CPU-ONLY: sycl-post-link
// CHK-CPU-ONLY-NOT: sycl-post-link

/// Objects providing device code for the same target: the archive device code
/// is added to the same image.
// RUN: clang-linker-wrapper --dry-run --linker-path=/usr/bin/ld -o /dev/null %t.o %t.a 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-ONE-IMAGE %s
// CHK-ONE-IMAGE: sycl-post-link
// CHK-ONE-IMAGE-NOT: sycl-post-link

/// Host only objects without archives: no device link.
// RUN: clang-linker-wrapper --dry-run --linker-path=/usr/bin/ld -o /dev/null %t_host.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-EXTRACT %s
// CHK-NO-EXTRACT-NOT: sycl-post-link

void foo() {}
