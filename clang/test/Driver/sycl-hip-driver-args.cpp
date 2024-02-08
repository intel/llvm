// Test that driver has the correct flags for SYCL HIP compilation

// RUN: %clangxx -### %s -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx1031 -fno-sycl-libspirv -nogpulib --sysroot=%S/Inputs/SYCL %s 2>&1 \
// RUN: | FileCheck %s

// CHECK: sycl-post-link{{.*}} "-emit-program-metadata" {{.*}}
// CHECK-NOT: "-cc1"{{.*}}"-fvisibility=hidden"

