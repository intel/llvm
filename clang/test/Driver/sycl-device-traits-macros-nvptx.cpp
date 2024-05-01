// REQUIRES: nvptx-registered-target

// Check device traits macros are defined if sycl is enabled:
// RUN:   %clangxx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA %s
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA: "-D__SYCL_ANY_DEVICE_HAS_{{.*}}__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"

