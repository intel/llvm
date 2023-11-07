// Tests for -fgpu-rdc with CUDA

// REQUIRES: nvptx-registered-target

// UNSUPPORTED: system-windows

// RUN: %clangxx -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_61 -fgpu-rdc -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -check-prefix=CHECK-SYCL_RDC_NVPTX

// Verify that ptxas does not pass "-c"
// CHECK-SYCL_RDC_NVPTX: {{.*}} "-cc1" "-triple" "nvptx64-nvidia-cuda" {{.*}} "-target-cpu" "sm_61" "-target-feature" "+ptx{{[0-9]+}}" {{.*}} "-fgpu-rdc" {{.*}} "-o" "[[PTX:.+]].s"
// CHECK-SYCL_RDC_NVPTX-NOT: ptxas{{.*}}"-m64" "-O3" "--gpu-name" "sm_61" "--output-file" "[[CUBIN:.+]].cubin" "[[PTX]].s" "-c"
