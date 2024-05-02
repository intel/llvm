// Check device traits macros are defined if sycl is enabled:
// Compiling for the default CUDA target passing the triple to '-fsycl-targets'.
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvptx64-nvidia-cuda -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE %s
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"

// Check device traits macros are defined if sycl is enabled:
// Compiling for a specific CUDA target passing the device to '-fsycl-targets'.
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvidia_gpu_sm_80 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE %s
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"

// Check device traits macros are defined if sycl is enabled:
// Compiling for a CUDA target passing the device arch to '--offload-arch'.
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_80 -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH %s
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH: "{{(-D__SYCL_ALL_DEVICES_HAVE_)?}}{{.*}}{{(__=1)?}}"
