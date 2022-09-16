/// Diagnose unsupported options specific to SYCL compilations
// RUN: %clangxx -fsycl -fsanitize=address -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s --check-prefix=SANITIZE -DARCH=spir64
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -fsanitize=address -### %s 2>&1 \
// RUN:  | FileCheck %s --check-prefix=SANITIZE -DARCH=spir64_gen
// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga -fsanitize=address -### %s 2>&1 \
// RUN:  | FileCheck %s --check-prefix=SANITIZE -DARCH=spir64_fpga
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fsanitize=address -### %s 2>&1 \
// RUN:  | FileCheck %s --check-prefix=SANITIZE -DARCH=spir64_x86_64
// SANITIZE: ignoring '-fsanitize=address' option as it is not currently supported for target '[[ARCH]]{{.*}}' [-Woption-ignored]
