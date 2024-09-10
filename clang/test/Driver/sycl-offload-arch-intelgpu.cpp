/// Tests the behaviors of using -fsycl --offload-new-driver 
//  --offload-arch=<intel-gpu-values>.

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=bdw %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE,CLANG-OFFLOAD-PACKAGER -DDEV_STR=bdw -DMAC_STR=BDW

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=cfl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE,CLANG-OFFLOAD-PACKAGER -DDEV_STR=cfl -DMAC_STR=CFL


///If Arch is icl, map it to icllp internally to create D__SYCL_TARGET_INTEL_GPU_

// TARGET-TRIPLE: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// TARGET-TRIPLE: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER: clang-offload-packager{{.*}} "--image={{.*}}triple=spir64_gen-unknown-unknown,arch=[[DEV_STR]],kind=sycl"


