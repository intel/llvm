// Check device traits macros are defined if sycl is enabled:
// Compiling for a specific HIP target passing the device to '-fsycl-targets'.
// RUN: %clangxx -nogpulib -fsycl -fsycl-targets=amd_gpu_gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE %s
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE-NOT: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE: "-D__SYCL_ANY_DEVICE_HAS_{{.*}}__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-DEVICE-TRIPLE: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"

// Check device traits macros are defined if sycl is enabled:
// Compiling for a HIP target passing the device arch to '--offload-arch'.
// RUN: %clangxx -nogpulib -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH %s
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH-NOT: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH: "-D__SYCL_ANY_DEVICE_HAS_{{.*}}__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-OFFLOAD-ARCH: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"
