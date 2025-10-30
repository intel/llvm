/// Test that SYCL bitcode device libraries are properly separated for NVIDIA and AMD targets

/// Check devicelib and libspirv are linked for nvptx
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-BC %s

// CHECK-NVPTX-BC: clang-linker-wrapper
// CHECK-NVPTX-BC-SAME: "--bitcode-library=nvptx64-nvidia-cuda={{.*}}devicelib-nvptx64-nvidia-cuda.bc" "--bitcode-library=nvptx64-nvidia-cuda={{.*}}libspirv-nvptx64-nvidia-cuda.bc"

/// Check devicelib is linked for amdgcn
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-AMD-BC %s

// CHECK-AMD-BC: clang-linker-wrapper
// CHECK-AMD-BC-SAME: "--bitcode-library=amdgcn-amd-amdhsa={{.*}}devicelib-amdgcn-amd-amdhsa.bc"

/// Check linking with multiple targets
// RUN: %clang -### -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MULTI-TARGET %s

// CHECK-MULTI-TARGET: clang-linker-wrapper
// CHECK-MULTI-TARGET-SAME: "--bitcode-library=amdgcn-amd-amdhsa={{.*}}devicelib-amdgcn-amd-amdhsa.bc" "--bitcode-library=nvptx64-nvidia-cuda={{.*}}devicelib-nvptx64-nvidia-cuda.bc" "--bitcode-library=nvptx64-nvidia-cuda={{.*}}libspirv-nvptx64-nvidia-cuda.bc"

/// Test --bitcode-library with nvptx dummy libraries
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.nvptx.devicelib.bc
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.nvptx.libspirv.bc
// RUN: %clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda --offload-new-driver -c %s -o %t.nvptx.o -nocudalib
// RUN: clang-linker-wrapper --bitcode-library=nvptx64-nvidia-cuda=%t.nvptx.devicelib.bc --bitcode-library=nvptx64-nvidia-cuda=%t.nvptx.libspirv.bc \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.nvptx.o -o a.out 2>&1 | FileCheck -check-prefix=CHECK-WRAPPER-NVPTX %s

// CHECK-WRAPPER-NVPTX: llvm-link{{.*}} {{.*}}.nvptx.devicelib.bc {{.*}}.nvptx.libspirv.bc

/// Test --bitcode-library with amdgcn dummy library
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amd.devicelib.bc
// RUN: %clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 --offload-new-driver -c %s -o %t.amd.o -nogpulib
// RUN: clang-linker-wrapper --bitcode-library=amdgcn-amd-amdhsa=%t.amd.devicelib.bc \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.amd.o -o a.out 2>&1 | FileCheck -check-prefix=CHECK-WRAPPER-AMD %s

// CHECK-WRAPPER-AMD: llvm-link{{.*}} {{.*}}.amd.devicelib.bc

/// Test --bitcode-library with multi-target bc libraries
// RUN: %clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --offload-new-driver -c %s -o %t.multi.o -nocudalib -nogpulib
// RUN: clang-linker-wrapper --bitcode-library=amdgcn-amd-amdhsa=%t.amd.devicelib.bc --bitcode-library=nvptx64-nvidia-cuda=%t.nvptx.devicelib.bc --bitcode-library=nvptx64-nvidia-cuda=%t.nvptx.libspirv.bc \
// RUN:   --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.multi.o -o a.out 2>&1 | FileCheck -check-prefix=CHECK-WRAPPER-MULTI %s

// CHECK-WRAPPER-MULTI: llvm-link{{.*}} {{.*}}.amd.devicelib.bc
// CHECK-WRAPPER-MULTI: llvm-link{{.*}} {{.*}}.nvptx.devicelib.bc {{.*}}.nvptx.libspirv.bc
