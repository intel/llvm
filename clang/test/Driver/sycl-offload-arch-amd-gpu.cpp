// SYCL AOT compilation to AMD GPUs using --offload-arch and --offload-new-driver

// AMD GPUs

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx700 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx700 -DMAC_STR=GFX700 -DTRIPLE_STR=amdgpu7.00

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx701 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx701 -DMAC_STR=GFX701 -DTRIPLE_STR=amdgpu7.01

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx702 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx702 -DMAC_STR=GFX702 -DTRIPLE_STR=amdgpu7.02

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx801 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx801 -DMAC_STR=GFX801 -DTRIPLE_STR=amdgpu8.01

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx802 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx802 -DMAC_STR=GFX802 -DTRIPLE_STR=amdgpu8.02

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx803 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx803 -DMAC_STR=GFX803 -DTRIPLE_STR=amdgpu8.03

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx805 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx805 -DMAC_STR=GFX805 -DTRIPLE_STR=amdgpu8.05

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx810 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx810 -DMAC_STR=GFX810 -DTRIPLE_STR=amdgpu8.10

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx900 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx900 -DMAC_STR=GFX900 -DTRIPLE_STR=amdgpu9.00

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx902 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx902 -DMAC_STR=GFX902 -DTRIPLE_STR=amdgpu9.02

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx904 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx904 -DMAC_STR=GFX904 -DTRIPLE_STR=amdgpu9.04

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx906 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx906 -DMAC_STR=GFX906 -DTRIPLE_STR=amdgpu9.06

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx908 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx908 -DMAC_STR=GFX908 -DTRIPLE_STR=amdgpu9.08

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx909 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx909 -DMAC_STR=GFX909 -DTRIPLE_STR=amdgpu9.09

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx90a -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx90a -DMAC_STR=GFX90A -DTRIPLE_STR=amdgpu9.0a

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx90c -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx90c -DMAC_STR=GFX90C -DTRIPLE_STR=amdgpu9.0c

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx942 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx942 -DMAC_STR=GFX942 -DTRIPLE_STR=amdgpu9.42

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1010 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1010 -DMAC_STR=GFX1010 -DTRIPLE_STR=amdgpu10.10

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1011 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1011 -DMAC_STR=GFX1011 -DTRIPLE_STR=amdgpu10.11

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1012 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1012 -DMAC_STR=GFX1012 -DTRIPLE_STR=amdgpu10.12

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1013 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1013 -DMAC_STR=GFX1013 -DTRIPLE_STR=amdgpu10.13

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1030 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1030 -DMAC_STR=GFX1030 -DTRIPLE_STR=amdgpu10.30

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1031 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1031 -DMAC_STR=GFX1031 -DTRIPLE_STR=amdgpu10.31

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1032 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1032 -DMAC_STR=GFX1032 -DTRIPLE_STR=amdgpu10.32

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1033 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1033 -DMAC_STR=GFX1033 -DTRIPLE_STR=amdgpu10.33

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1034 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1034 -DMAC_STR=GFX1034 -DTRIPLE_STR=amdgpu10.34

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1035 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1035 -DMAC_STR=GFX1035 -DTRIPLE_STR=amdgpu10.35

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1036 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1036 -DMAC_STR=GFX1036 -DTRIPLE_STR=amdgpu10.36

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1100 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1100 -DMAC_STR=GFX1100 -DTRIPLE_STR=amdgpu11.00

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1101 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1101 -DMAC_STR=GFX1101 -DTRIPLE_STR=amdgpu11.01

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1102 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1102 -DMAC_STR=GFX1102 -DTRIPLE_STR=amdgpu11.02

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1103 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1103 -DMAC_STR=GFX1103 -DTRIPLE_STR=amdgpu11.03

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1150 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1150 -DMAC_STR=GFX1150 -DTRIPLE_STR=amdgpu11.50

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1151 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1151 -DMAC_STR=GFX1151 -DTRIPLE_STR=amdgpu11.51

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1200 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1200 -DMAC_STR=GFX1200 -DTRIPLE_STR=amdgpu12.00

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1201 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1201 -DMAC_STR=GFX1201 -DTRIPLE_STR=amdgpu12.01

// TARGET-TRIPLE-AMD-GPU: clang{{.*}} "-triple" "[[TRIPLE_STR]]-amd-amdhsa"
// TARGET-TRIPLE-AMD-GPU: "-D__SYCL_TARGET_AMD_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER-AMD: llvm-offload-binary{{.*}} "--image={{.*}}triple=[[TRIPLE_STR]]-amd-amdhsa,arch=[[DEV_STR]],kind=sycl"

// Tests for handling an invalid architecture.
//
// RUN: not %clangxx --offload-new-driver -fsycl --offload-arch=gfx10_3_generic %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=ERROR %s
// RUN: not %clang_cl --offload-new-driver -fsycl --offload-arch=gfx10_3_generic %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=ERROR %s

// ERROR: error: unsupported offload gpu architecture: gfx10_3_generic
