// REQUIRES: amdgpu-registered-target

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -nogpulib\
// RUN:   %S/Inputs/SYCL/libamdgcn-gfx908.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -nogpulib\
// RUN:   -L%S/Inputs/SYCL -lamdgcn-gfx908 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -nogpulib\
// RUN:   %S/Inputs/SYCL/objamdgcn-gfx908.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_DIAG
// AMDGCN_DIAG: linked binaries do not contain expected 'amdgcn-amd-amdhsa-gfx906' target; found targets: 'amdgcn-amd-amdhsa-gfx908' [-Wsycl-target]

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib\
// RUN:   %S/Inputs/SYCL/libamdgcn-gfx908.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib\
// RUN:   -L%S/Inputs/SYCL -lamdgcn-gfx908 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib\
// RUN:   %S/Inputs/SYCL/objamdgcn-gfx908.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -nogpulib\
// RUN:   -Wno-sycl-target %S/Inputs/SYCL/objamdgcn-gfx908.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=AMDGCN_MATCH_DIAG
// AMDGCN_MATCH_DIAG-NOT: linked binaries do not contain expected
