/// Tests the behaviors of using -fsycl --offload-new-driver 
//  --offload-arch=<intel-gpu-values>.

// SYCL AOT compilation to Intel GPUs using --offload-arch

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=bmg_g21 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=bdw -DMAC_STR=BDW



// TARGET-TRIPLE-GPU: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// TARGET-TRIPLE-GPU: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER-GPU: clang-offload-packager{{.*}} "--image={{.*}}triple=spir64_gen-unknown-unknown,arch=[[DEV_STR]],kind=sycl"
// CLANG-OFFLOAD-PACKAGER-GPU-OPTS: clang-offload-packager{{.*}} "--image={{.*}}triple=spir64_gen-unknown-unknown,arch=[[DEV_STR]],kind=sycl{{.*}}"

