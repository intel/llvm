/// Diagnose unsupported options specific to SYCL compilations
// RUN: %clangxx -fsycl -fsanitize=address -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fsanitize=address
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -fsanitize=address -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_gen -DOPT=-fsanitize=address
// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga -fsanitize=address -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_fpga -DOPT=-fsanitize=address
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fsanitize=address -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_x86_64 -DOPT=-fsanitize=address

// RUN: %clangxx -fsycl -fcf-protection -fsycl-targets=spir64 -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64 -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_gen -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_fpga -DOPT=-fcf-protection
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -fcf-protection -### %s 2>&1 \
// RUN:  | FileCheck %s -DARCH=spir64_x86_64 -DOPT=-fcf-protection

// CHECK: ignoring '[[OPT]]' option as it is not currently supported for target '[[ARCH]]{{.*}}' [-Woption-ignored]
