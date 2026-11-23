// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=intel_gpu_cri %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=intel_gpu_35_11_0 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=intel_gpu_bmg_g31 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-NON-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=spir64 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-NON-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen -device \
// RUN:   -Xsycl-target-backend=spir64_gen cri %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen -device \
// RUN:   -Xsycl-target-backend=spir64_gen bmg_g31 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-NON-CRI

// CHECK-CRI: llvm-spirv{{.*}} "-spirv-ext=-all{{.*}},+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics"
// CHECK-NON-CRI: llvm-spirv{{.*}} "-spirv-ext=-all
// CHECK-NON-CRI-NOT: SPV_KHR_bfloat16
// CHECK-NON-CRI-NOT: SPV_INTEL_16bit_atomics

/// Same checks when using the SPIR-V backend (llc) instead of llvm-spirv.
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-use-spirv-backend-for-spirv-gen \
// RUN:   -fsycl-targets=intel_gpu_cri %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-BE-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-use-spirv-backend-for-spirv-gen \
// RUN:   -fsycl-targets=intel_gpu_bmg_g31 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-BE-NON-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-use-spirv-backend-for-spirv-gen \
// RUN:   -fsycl-targets=spir64 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-BE-NON-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-use-spirv-backend-for-spirv-gen \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen -device \
// RUN:   -Xsycl-target-backend=spir64_gen cri %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-BE-CRI
// RUN: %clang -target x86_64-unknown-linux-gnu -fsycl --no-offload-new-driver \
// RUN:   -fsycl-use-spirv-backend-for-spirv-gen \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen -device \
// RUN:   -Xsycl-target-backend=spir64_gen bmg_g31 %s -### 2>&1 \
// RUN:  | FileCheck %s -check-prefix=CHECK-BE-NON-CRI

// CHECK-BE-CRI: llc{{.*}} "-spirv-ext={{.*}},+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics"
// CHECK-BE-NON-CRI: llc{{.*}} "-spirv-ext=
// CHECK-BE-NON-CRI-NOT: SPV_KHR_bfloat16
// CHECK-BE-NON-CRI-NOT: SPV_INTEL_16bit_atomics
