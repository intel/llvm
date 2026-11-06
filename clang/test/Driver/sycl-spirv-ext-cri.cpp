/// Check that SPV_KHR_bfloat16 and SPV_INTEL_16bit_atomics are enabled by
/// default for CRI AOT only.

// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=intel_gpu_cri %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=intel_gpu_bmg_g31 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-NON-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=spir64 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-NON-CRI

// CHECK-CRI: llvm-spirv{{.*}} "-spirv-ext=-all{{.*}},+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics"
// CHECK-NON-CRI: llvm-spirv{{.*}} "-spirv-ext=-all
// CHECK-NON-CRI-NOT: SPV_KHR_bfloat16
// CHECK-NON-CRI-NOT: SPV_INTEL_16bit_atomics
