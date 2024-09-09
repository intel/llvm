/// Tests the behaviors of using -fsycl --offload-new-driver 
//  --offload-arch=<intel-gpu-values>.

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=bdw %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=bdw -DMAC_STR=BDW
// MACRO: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// MACRO: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// MACRO: clang{{.*}} "-fsycl-is-host"
// MACRO: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// DEVICE: ocloc{{.*}} "-device" "[[DEV_STR]]"
