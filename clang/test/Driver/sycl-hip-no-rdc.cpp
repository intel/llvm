// Test for -fno-sycl-rdc with HIP to make sure sycl-post-link is marked as outputting ir

// RUN: touch %t1.cpp
// RUN: %clang -### -fsycl -fno-sycl-rdc -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1031 --sysroot=%S/Inputs/SYCL %t1.cpp 2>&1 -ccc-print-phases | FileCheck %s

// CHECK: sycl-post-link, {{{.*}}}, ir, (device-sycl, gfx1031)
