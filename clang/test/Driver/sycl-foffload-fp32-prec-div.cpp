// Test SYCL -foffload-fp32-prec-div

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:    -fsycl-targets=spir64_gen -foffload-fp32-prec-div %s 2>&1 \
// RUN:   | FileCheck -check-prefix=AOT %s

// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -foffload-fp32-prec-div %s 2>&1 \
// RUN:   | FileCheck -check-prefix=JIT %s

// AOT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64_gen-unknown-unknown,arch={{.*}},kind=sycl,compile-opts=-options -ze-fp32-correctly-rounded-divide-sqrt{{.*}}"

// JIT: clang-offload-packager{{.*}} "--image=file={{.*}}.bc,triple=spir64-unknown-unknown,arch={{.*}}compile-opts={{.*}}-foffload-fp32-prec-div"
