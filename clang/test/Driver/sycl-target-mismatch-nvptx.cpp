// REQUIRES: nvptx-registered-target

// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60  \
// RUN:   %S/Inputs/SYCL/libnvptx64-sm_50.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_DIAG
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 \
// RUN:   -L%S/Inputs/SYCL -lnvptx64-sm_50 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_DIAG
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 \
// RUN:   %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_DIAG
// NVPTX64_DIAG: linked binaries do not contain expected 'nvptx64-nvidia-cuda-sm_60' target; found targets: 'nvptx64-nvidia-cuda-sm_50' [-Wsycl-target]

// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:   %S/Inputs/SYCL/libnvptx64-sm_50.a -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:   -L%S/Inputs/SYCL -lnvptx64-sm_50 -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 \
// RUN:   %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 \
// RUN:   -Wno-sycl-target %S/Inputs/SYCL/objnvptx64-sm_50.o -### %s 2>&1 \
// RUN:  | FileCheck %s -check-prefix=NVPTX64_MATCH_DIAG
// NVPTX64_MATCH_DIAG-NOT: linked binaries do not contain expected
