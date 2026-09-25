/// Test that SYCL bitcode device libraries are properly separated for NVIDIA and AMD targets.

/// Check devicelib are linked for nvptx.
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-BC %s

// RUN: %clang_cl -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-BC %s

// CHECK-NVPTX-BC: clang-linker-wrapper
// CHECK-NVPTX-BC-SAME: "--bitcode-library=nvptx64-nvidia-cuda={{.*}}devicelib-nvptx64-nvidia-cuda.bc"

/// Check devicelib is linked for amdgcn.
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-AMD-BC %s

// RUN: %clang_cl -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-AMD-BC %s

// CHECK-AMD-BC: clang-linker-wrapper
// CHECK-AMD-BC-SAME: "--bitcode-library=amdgcn-amd-amdhsa={{.*}}devicelib-amdgcn-amd-amdhsa.bc"

/// Check linking with multiple targets.
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MULTI-TARGET %s

// RUN: %clang_cl -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MULTI-TARGET %s

// CHECK-MULTI-TARGET: clang-linker-wrapper
// CHECK-MULTI-TARGET-SAME: "--bitcode-library=amdgcn-amd-amdhsa={{.*}}devicelib-amdgcn-amd-amdhsa.bc" "--bitcode-library=nvptx64-nvidia-cuda={{.*}}devicelib-nvptx64-nvidia-cuda.bc"
