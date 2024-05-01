// REQUIRES: amdgpu-registered-target

// Check device traits macros are defined if sycl is enabled:
// RUN:   %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-AMDGCN-AMD-AMDHSA %s
// CHECK-SYCL-AMDGCN-AMD-AMDHSA-NOT: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA: "-D__SYCL_ANY_DEVICE_HAS_{{.*}}__=1"
// CHECK-SYCL-AMDGCN-AMD-AMDHSA: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"

