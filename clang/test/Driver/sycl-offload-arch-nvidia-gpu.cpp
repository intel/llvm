/// Tests the behaviors of using --offload-arch for offloading
// SYCL kernels to NVidia GPUs using --offload-new-driver.

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_50 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_50 -DMAC_STR=SM_50

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_52 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_52 -DMAC_STR=SM_52

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_53 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_53 -DMAC_STR=SM_53

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_60 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_60 -DMAC_STR=SM_60

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_61 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_61 -DMAC_STR=SM_61

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_62 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_62 -DMAC_STR=SM_62

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_70 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_70 -DMAC_STR=SM_70

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_72 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_72 -DMAC_STR=SM_72

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_75 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_75 -DMAC_STR=SM_75

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_80 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_80 -DMAC_STR=SM_80

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_86 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_86 -DMAC_STR=SM_86

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_87 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_87 -DMAC_STR=SM_87

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_89 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_89 -DMAC_STR=SM_89

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_90 -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_90 -DMAC_STR=SM_90

// RUN: %clangxx --offload-new-driver -fsycl --offload-arch=sm_90a -nocudalib -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CLANG-OFFLOAD-PACKAGER-GPU,MACRO_NVIDIA -DDEV_STR=sm_90a -DMAC_STR=SM_90A

// MACRO_NVIDIA: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"
// MACRO_NVIDIA: "-D__SYCL_TARGET_NVIDIA_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER-GPU: clang-offload-packager{{.*}} "--image={{.*}}triple=nvptx64-nvidia-cuda,arch=[[DEV_STR]],kind=sycl"

