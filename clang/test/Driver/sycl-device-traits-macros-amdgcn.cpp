// Compiling for a specific HIP target passing the device to '-fsycl-targets'.
//
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx700 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx701 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx702 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx703 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx704 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx705 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx801 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx802 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx803 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx805 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx810 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx900 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx902 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx904 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx908 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx909 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx90a \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx90c \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx940 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx941 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx942 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1010 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1011 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1012 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1013 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1030 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1031 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1032 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1033 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1034 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1035 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1036 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1100 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1101 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1102 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1103 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1150 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1151 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE

// Compiling for a HIP target passing the device arch to '--offload-arch'.
//
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx700 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx701 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx702 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx703 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx704 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx705 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx801 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx802 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx803 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx805 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx810 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx900 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx902 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx904 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx909 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90c \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx940 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx941 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx942 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1010 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1011 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1012 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1013 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1030 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1031 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1032 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1033 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1034 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1035 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1036 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1100 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1101 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1102 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1103 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1150 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx1151 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH

// Check device traits macros are defined if sycl is enabled:

// Test -fsycl-targets device configuration aspects are parsed correctly.
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE-NOT: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE: "-D__SYCL_ANY_DEVICE_HAS_[[ASPECT:[a-z0-9_]+]]__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE: "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT]]__=1"

// Test --offload-arch device configuration aspects are parsed correctly.
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH-NOT: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH: "-D__SYCL_ANY_DEVICE_HAS_[[ASPECT:[a-z0-9_]+]]__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH: "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT]]__=1"
