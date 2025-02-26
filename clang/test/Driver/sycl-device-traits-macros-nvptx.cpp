// Check device traits macros are defined if sycl is enabled:
// Compiling for the default CUDA target passing the triple to '-fsycl-targets'.
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvptx64-nvidia-cuda -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE %s
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE-NOT: "-D__SYCL_ANY_DEVICE_HAS_{{[a-z0-9_]+}}__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEFAULT-TRIPLE: "-D__SYCL_ALL_DEVICES_HAVE_{{[a-z0-9_]+}}__=1"

// Compiling for a specific CUDA target passing the device to '-fsycl-targets'.
//
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_50 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_52 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_53 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_60 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_61 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_62 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_70 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_72 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_75 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_80 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_86 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_87 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_89 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_90 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_90a -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE

// Compiling for a CUDA target passing the device arch to '--offload-arch' (using the '--cuda-gpu-arch' alias).
//
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_50 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_52 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_53 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_61 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_62 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_72 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_75 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_86 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_87 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_89 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_90 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_90a -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH

// Check device traits macros are defined if sycl is enabled:

// Test -fsycl-targets device configuration aspects are parsed correctly.
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE-NOT: "-D__SYCL_ANY_DEVICE_HAS_{{[a-z0-9_]+}}__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-DEVICE-TRIPLE: "-D__SYCL_ALL_DEVICES_HAVE_{{[a-z0-9_]+}}__=1"

// Test --offload-arch device configuration aspects are parsed correctly.
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH-NOT: "-D__SYCL_ANY_DEVICE_HAS_{{[a-z0-9_]+}}__=1"
// CHECK-SYCL-NVPTX-NVIDIA-CUDA-OFFLOAD-ARCH: "-D__SYCL_ALL_DEVICES_HAVE_{{[a-z0-9_]+}}__=1"

// Test all Cuda devices support at least the aspects of the least capable one.
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvidia_gpu_sm_50 -### %s > %t 2>&1
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvidia_gpu_sm_60 -### %s >> %t 2>&1
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvidia_gpu_sm_70 -### %s >> %t 2>&1
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvidia_gpu_sm_80 -### %s >> %t 2>&1
// RUN: %clangxx -nocudalib -fsycl -fsycl-targets=nvidia_gpu_sm_90 -### %s >> %t 2>&1
// RUN: FileCheck --input-file=%t --check-prefixes=CHECK-SM50,CHECK-SM60,CHECK-SM70,CHECK-SM80,CHECK-SM90 %s
// CHECK-SM50: "-D__SYCL_TARGET_NVIDIA_GPU_SM_50__"{{.*}} "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT:[a-z0-9]+]]__=1"
// CHECK-SM60: "-D__SYCL_TARGET_NVIDIA_GPU_SM_60__"{{.*}} "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT]]__=1"
// CHECK-SM70: "-D__SYCL_TARGET_NVIDIA_GPU_SM_70__"{{.*}} "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT]]__=1"
// CHECK-SM80: "-D__SYCL_TARGET_NVIDIA_GPU_SM_80__"{{.*}} "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT]]__=1"
// CHECK-SM90: "-D__SYCL_TARGET_NVIDIA_GPU_SM_90__"{{.*}} "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECT]]__=1"
