// REQUIRES: clang-driver
// REQUIRES: amdgpu-registered-target
// REQUIRES: !system-windows

// RUN: %clang -### \
// RUN:   -fsycl -fsycl-targets=amdgcn-amd-amdhsa     \
// RUN:   -Xsycl-target-backend --offload-arch=gfx900 \
// RUN:   -fsycl-fp32-prec-sqrt \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s \
// RUN: 2>&1 | FileCheck  --check-prefix=CHECK-CORRECT %s

// CHECK-CORRECT: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc"

// RUN: %clang -### \
// RUN:   -fsycl -fsycl-targets=amdgcn-amd-amdhsa     \
// RUN:   -Xsycl-target-backend --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s \
// RUN: 2>&1 | FileCheck  --check-prefix=CHECK-APPROX %s

// CHECK-APPROX: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc"

// RUN: %clang -### \
// RUN:   -fsycl -fsycl-targets=amdgcn-amd-amdhsa     \
// RUN:   -Xsycl-target-backend --offload-arch=gfx900 \
// RUN:   -fsycl-fp32-prec-sqrt -fno-hip-fp32-correctly-rounded-divide-sqrt \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s \
// RUN: 2>&1 | FileCheck  --check-prefix=CHECK-CONFLICT %s

// CHECK-CONFLICT: warning: argument unused during compilation: '-fno-hip-fp32-correctly-rounded-divide-sqrt'
// CHECK-CONFLICT: "-mlink-builtin-bitcode" "{{.*}}/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc"

void func(){};
