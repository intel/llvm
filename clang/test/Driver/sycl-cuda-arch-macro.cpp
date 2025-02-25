// Verify the __CUDA_ARCH__ macro has not been defined when offloading SYCL on NVPTX
// RUN: %clangxx -E -dM -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_80 -nocudalib -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-CUDA-ARCH-MACRO %s
// Verify the __CUDA_ARCH__ macro has not been defined when offloading SYCL on AMDGPU
// RUN: %clangxx -E -dM -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a -nogpulib -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-CUDA-ARCH-MACRO %s
// CHECK-CUDA-ARCH-MACRO-NOT: #define __CUDA_ARCH__ {{[0-9]+}}

// Verify that '-fcuda-is-device' is not supplied when offloading SYCL on NVPTX
// NOTE: AMDGPU targets, i.e. "amdgcn-amd-amdhsa" may rely on "fcuda-is-device"
// RUN: %clangxx -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_80 -nocudalib -fno-sycl-libspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-CUDA-IS-DEVICE %s
// CHECK-CUDA-IS-DEVICE: clang{{.*}} "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CHECK-CUDA-IS-DEVICE-NOT: "-fcuda-is-device"
// CHECK-CUDA-IS-DEVICE-SAME: "-fsycl-is-device"
