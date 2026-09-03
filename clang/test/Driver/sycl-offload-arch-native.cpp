/// Tests the behavior of using -fsycl --offload-new-driver
//  --offload-arch=native.
//
// SYCL offloading is supported for Intel GPU devices only, so the driver is
// expected to restrict the 'offload-arch' utility to the Intel devices.

// Needs chmod
// UNSUPPORTED: system-windows

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/offload-arch/offload_arch_only_intel %t/
// RUN: echo '#!/bin/sh' > %t/offload_arch_empty
// RUN: chmod +x %t/offload_arch_only_intel %t/offload_arch_empty

// The devices of the other vendors reported by 'offload-arch' are not turned
// into SYCL offloading targets.
// RUN: %clangxx -### --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_only_intel %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NATIVE \
// RUN:     --implicit-check-not="nvptx64-nvidia-cuda" \
// RUN:     --implicit-check-not="amdgcn-amd-amdhsa"

// NATIVE: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// NATIVE: "-D__SYCL_TARGET_INTEL_GPU_BMG_G21__"
// NATIVE: llvm-offload-binary{{.*}} "--image={{.*}}triple=spir64_gen-unknown-unknown,arch=bmg_g21,kind=sycl{{.*}}"

// Case when no device supporting SYCL is detected in the system.
// RUN: not %clangxx -### --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_empty %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NO-DEVICE

// NO-DEVICE: error: cannot determine sycl architecture: No GPU detected in the system
