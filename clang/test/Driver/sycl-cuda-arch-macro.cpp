// RUN: %clangxx %s -fsycl -nocudalib -fno-sycl-libspirv -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_80 -E -dM \
// RUN: | FileCheck --check-prefix=CHECK-CUDA-ARCH-MACRO %s
// CHECK-CUDA-ARCH-MACRO-NOT: #define __CUDA_ARCH__ {{[0-9]+}}
