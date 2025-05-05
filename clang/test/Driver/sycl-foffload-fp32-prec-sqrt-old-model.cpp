// Test SYCL -foffload-fp32-prec-sqrt

// RUN: %clang -### -fsycl --no-offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -foffload-fp32-prec-sqrt %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AOT %s

// RUN: %clang -### -fsycl --no-offload-new-driver \
// RUN:   -foffload-fp32-prec-sqrt %s 2>&1 \
// RUN:   | FileCheck -check-prefix=JIT %s

// AOT: "-ze-fp32-correctly-rounded-divide-sqrt"

// JIT: clang-offload-wrapper{{.*}} "-compile-opts={{.*}}-foffload-fp32-prec-sqrt"
