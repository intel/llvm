/// Test that SYCL bitcode device libraries are properly separated for NVIDIA and AMD targets.

// FIXME: Force linux targets to allow for the libraries to be found.  Dummy
// inputs for --sysroot should be updated to work better for Windows.

/// Check devicelib are linked for nvptx.
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --sysroot=%S/Inputs/SYCL \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-BC %s

// TODO: Check the clang device for the device library
// CHECK-NVPTX-BC: clang{{.*}}

/// Check devicelib is linked for amdgcn.
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   --sysroot=%S/Inputs/SYCL \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-AMD-BC %s

// TODO: Check the clang device for the device library
// CHECK-AMD-BC: clang{{.*}}

/// Check linking with multiple targets.
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   --sysroot=%S/Inputs/SYCL \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MULTI-TARGET %s

// TODO: Check the clang device for the device library
// CHECK-MULTI-TARGET: clang{{.*}}
