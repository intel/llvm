/// Check for diagnostic when command line link targets to not match object
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/liblin64.a \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_GEN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -L%S/Inputs/SYCL -llin64 \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_GEN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/objlin64.o \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_GEN_DIAG
// SPIR64_GEN_DIAG: linked binaries do not contain expected 'spir64_gen-unknown-unknown' target; found targets: 'spir64-unknown-unknown{{.*}}, spir64-unknown-unknown{{.*}}' [-Wsycl-target]

// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/SYCL/liblin64.a \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -L%S/Inputs/SYCL -llin64 \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64 %S/Inputs/SYCL/objlin64.o \
// RUN:   -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/SYCL/liblin64.a \
// RUN:   -Wno-sycl-target -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=SPIR64_DIAG
// SPIR64_DIAG-NOT: linked binaries do not contain expected

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60  \
// RUN:   %S/Inputs/SYCL/libnvptx64-sm_50.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 \
// RUN:   -L%S/Inputs/SYCL -lnvptx64-sm_50 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 \
// RUN:   %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_DIAG
// NVPTX64_DIAG: linked binaries do not contain expected 'nvptx64-nvidia-cuda-sm_60' target; found targets: 'nvptx64-nvidia-cuda-sm_50' [-Wsycl-target]

// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:   %S/Inputs/SYCL/libnvptx64-sm_50.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:   -L%S/Inputs/SYCL -lnvptx64-sm_50 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:   %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 \
// RUN:   -Wno-sycl-target %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// NVPTX64_MATCH_DIAG-NOT: linked binaries do not contain expected

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906  \
// RUN:   %S/Inputs/SYCL/libamdgcn-gfx908.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 \
// RUN:   -L%S/Inputs/SYCL -lamdgcn-gfx908 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 \
// RUN:   %S/Inputs/SYCL/objamdgcn-gfx908.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_DIAG
// AMDGCN_DIAG: linked binaries do not contain expected 'amdgcn-amd-amdhsa-gfx906' target; found targets: 'amdgcn-amd-amdhsa-gfx908' [-Wsycl-target]

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 \
// RUN:   %S/Inputs/SYCL/libamdgcn-gfx908.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 \
// RUN:   -L%S/Inputs/SYCL -lamdgcn-gfx908 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 \
// RUN:   %S/Inputs/SYCL/objamdgcn-gfx908.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 \
// RUN:   -Wno-sycl-target %S/Inputs/SYCL/objamdgcn-gfx908.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// AMDGCN_MATCH_DIAG-NOT: linked binaries do not contain expected
