// SYCL AOT compilation to AMD GPUs using --offload-arch and --offload-new-driver

// AMD GPUs

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx700 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx700 -DMAC_STR=GFX700

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx701 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx701 -DMAC_STR=GFX701

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx702 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx702 -DMAC_STR=GFX702

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx801 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx801 -DMAC_STR=GFX801

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx802 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx802 -DMAC_STR=GFX802

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx803 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx803 -DMAC_STR=GFX803

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx805 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx805 -DMAC_STR=GFX805

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx810 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx810 -DMAC_STR=GFX810

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx900 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx900 -DMAC_STR=GFX900

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx902 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx902 -DMAC_STR=GFX902

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx904 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx904 -DMAC_STR=GFX904

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx906 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx906 -DMAC_STR=GFX906

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx908 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx908 -DMAC_STR=GFX908

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx909 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx909 -DMAC_STR=GFX909

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx90a -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx90a -DMAC_STR=GFX90A

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx90c -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx90c -DMAC_STR=GFX90C

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx940 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx940 -DMAC_STR=GFX940

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx941 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx941 -DMAC_STR=GFX941

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx942 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx942 -DMAC_STR=GFX942

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1010 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1010 -DMAC_STR=GFX1010

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1011 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1011 -DMAC_STR=GFX1011

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1012 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1012 -DMAC_STR=GFX1012

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1013 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1013 -DMAC_STR=GFX1013

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1030 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1030 -DMAC_STR=GFX1030

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1031 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1031 -DMAC_STR=GFX1031

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1032 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1032 -DMAC_STR=GFX1032

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1033 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1033 -DMAC_STR=GFX1033

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1034 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1034 -DMAC_STR=GFX1034

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1035 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1035 -DMAC_STR=GFX1035

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1036 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1036 -DMAC_STR=GFX1036

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1100 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1100 -DMAC_STR=GFX1100

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1101 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1101 -DMAC_STR=GFX1101

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1102 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1102 -DMAC_STR=GFX1102

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1103 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1103 -DMAC_STR=GFX1103

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1150 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1150 -DMAC_STR=GFX1150

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1151 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1151 -DMAC_STR=GFX1151

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1200 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1200 -DMAC_STR=GFX1200

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=gfx1201 -nogpulib -fno-sycl-libspirv %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-AMD-GPU,CLANG-OFFLOAD-PACKAGER-AMD -DDEV_STR=gfx1201 -DMAC_STR=GFX1201

// TARGET-TRIPLE-AMD-GPU: clang{{.*}} "-triple" "amdgcn-amd-amdhsa"
// TARGET-TRIPLE-AMD-GPU: "-D__SYCL_TARGET_AMD_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER-AMD: clang-offload-packager{{.*}} "--image={{.*}}triple=amdgcn-amd-amdhsa,arch=[[DEV_STR]],kind=sycl"

// Tests for handling an invalid architecture.
//
// RUN: not %clangxx --offload-new-driver -fsycl --offload-arch=gfx10_3_generic %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=ERROR %s
// RUN: not %clang_cl --offload-new-driver -fsycl --offload-arch=gfx10_3_generic %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=ERROR %s

// ERROR: error: SYCL target is invalid: 'gfx10_3_generic'


