// Test SYCL -foffload-fp32-prec-sqrt

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:    -fsycl-targets=spir64_gen -foffload-fp32-prec-sqrt %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AOT %s

// RUN: %clang -### -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL \
// RUN:   -foffload-fp32-prec-sqrt %s 2>&1 \
// RUN:   | FileCheck -check-prefix=JIT %s

// AOT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch={{.*}},kind=sycl,compile-opts=-options -ze-fp32-correctly-rounded-divide-sqrt{{.*}}"

// JIT: llvm-offload-binary{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch={{.*}}compile-opts={{.*}}-foffload-fp32-prec-sqrt"
