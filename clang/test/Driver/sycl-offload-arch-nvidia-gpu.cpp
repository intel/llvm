/// Tests the behaviors of using --offload-arch for offloading
// SYCL kernels to NVidia GPUs using --offload-new-driver.

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_50 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_50 -DMAC_STR=SM_50

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_52 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_52 -DMAC_STR=SM_52

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_53 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_53 -DMAC_STR=SM_53

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_60 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_60 -DMAC_STR=SM_60

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_61 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_61 -DMAC_STR=SM_61

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_62 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_62 -DMAC_STR=SM_62

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_70 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_70 -DMAC_STR=SM_70

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_72 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_72 -DMAC_STR=SM_72

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_75 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_75 -DMAC_STR=SM_75

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_80 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_80 -DMAC_STR=SM_80

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_86 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_86 -DMAC_STR=SM_86

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_87 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_87 -DMAC_STR=SM_87

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_88 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_88 -DMAC_STR=SM_88

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_89 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_89 -DMAC_STR=SM_89

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_90 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_90 -DMAC_STR=SM_90

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_90a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_90a -DMAC_STR=SM_90A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_100 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_100 -DMAC_STR=SM_100

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_100a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_100a -DMAC_STR=SM_100A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_100f -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_100f -DMAC_STR=SM_100F

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_101 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_101 -DMAC_STR=SM_101

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_101a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_101a -DMAC_STR=SM_101A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_101f -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_101f -DMAC_STR=SM_101F

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_103 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_103 -DMAC_STR=SM_103

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_103a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_103a -DMAC_STR=SM_103A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_103f -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_103f -DMAC_STR=SM_103F

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_110 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_110 -DMAC_STR=SM_110

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_110a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_110a -DMAC_STR=SM_110A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_110f -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_110f -DMAC_STR=SM_110F

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_120 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_120 -DMAC_STR=SM_120

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_120a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_120a -DMAC_STR=SM_120A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_120f -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_120f -DMAC_STR=SM_120F

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_121 -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_121 -DMAC_STR=SM_121

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_121a -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_121a -DMAC_STR=SM_121A

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_121f -fno-sycl-libspirv -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_121f -DMAC_STR=SM_121F

// MACRO_NVIDIA: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"
// MACRO_NVIDIA: "-D__SYCL_TARGET_NVIDIA_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER-GPU: llvm-offload-binary{{.*}} "--image={{.*}}triple=nvptx64-nvidia-cuda,arch=[[DEV_STR]],kind=sycl"
