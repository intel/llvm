/// Test that SYCL bitcode device libraries are properly separated for NVIDIA and AMD targets.
/// Also test the SYCL device library location passed to clang-linker-wrapper
/// (-sycl-device-library-location) and sycl-post-link (--device-lib-dir).

/// Check that a device library which cannot be found is diagnosed.
// RUN: not %clang -### -fsycl --offload-new-driver -ccc-install-dir %t.nonexistent/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-MISSING %s
// CHECK-NVPTX-MISSING: cannot find expected SYCL device library 'devicelib-nvptx64-nvidia-cuda.bc'

/// Check devicelib are linked for nvptx.
// RUN: %clang -### -fsycl --offload-new-driver -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-BC %s

// RUN: %clang_cl -### -fsycl --offload-new-driver -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NVPTX-BC %s

// CHECK-NVPTX-BC: clang-linker-wrapper
// CHECK-NVPTX-BC-SAME: "--bitcode-library=nvptx64-nvidia-cuda={{.*}}devicelib-nvptx64-nvidia-cuda.bc"

/// Check devicelib is linked for amdgcn.
// RUN: %clang -### -fsycl --offload-new-driver -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-AMD-BC %s

// RUN: %clang_cl -### -fsycl --offload-new-driver -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-AMD-BC %s

// CHECK-AMD-BC: clang-linker-wrapper
// CHECK-AMD-BC-SAME: "--bitcode-library=amdgcn-amd-amdhsa={{.*}}devicelib-amdgcn-amd-amdhsa.bc"

/// Check linking with multiple targets.
// RUN: %clang -### -fsycl --offload-new-driver -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MULTI-TARGET %s

// RUN: %clang_cl -### -fsycl --offload-new-driver -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx900 \
// RUN:   --cuda-path=%S/Inputs/CUDA_102/usr/local/cuda \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-MULTI-TARGET %s

// CHECK-MULTI-TARGET: clang-linker-wrapper
// CHECK-MULTI-TARGET-SAME: "--bitcode-library=amdgcn-amd-amdhsa={{.*}}devicelib-amdgcn-amd-amdhsa.bc" "--bitcode-library=nvptx64-nvidia-cuda={{.*}}devicelib-nvptx64-nvidia-cuda.bc"

/// The device library location is resolved from the library being consumed.
/// Inputs/SYCL matches the Linux install layout, with the libraries in
/// lib/dpcpp-7/sycl. Make sure that location is found, rather than the
/// <driver>/../lib fallback.
/// Non-NativeCPU targets use the location of libsycl-native-bfloat16.bc.
// RUN: %clang -### -fsycl --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fsycl-targets=spir64 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-LIB-LOC-BF16 %s
// CHECK-LIB-LOC-BF16: clang-linker-wrapper{{.*}} "-sycl-device-library-location={{.*}}SYCL{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}dpcpp-7{{[/\\]+}}sycl"

// RUN: %clang -### -fsycl --no-offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fsycl-targets=spir64 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-LIB-LOC-BF16-POSTLINK %s
// CHECK-LIB-LOC-BF16-POSTLINK: sycl-post-link{{.*}} "--device-lib-dir={{.*}}SYCL{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}dpcpp-7{{[/\\]+}}sycl"

/// NativeCPU uses the location of libsycl-nativecpu_utils.bc.
// RUN: %clang -### -fsycl --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=native_cpu \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-LIB-LOC-NATIVECPU %s
// CHECK-LIB-LOC-NATIVECPU: clang-linker-wrapper
// CHECK-LIB-LOC-NATIVECPU-SAME: "--bitcode-library=native_cpu={{.*}}SYCL{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}dpcpp-7{{[/\\]+}}sycl{{[/\\]+}}libsycl-nativecpu_utils.bc"
// CHECK-LIB-LOC-NATIVECPU-SAME: "-sycl-device-library-location={{.*}}SYCL{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}dpcpp-7{{[/\\]+}}sycl"

/// Mixed NativeCPU and SPIR targets also use the location of
/// libsycl-nativecpu_utils.bc.
// RUN: %clang -### -fsycl --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -ccc-install-dir %S/Inputs/SYCL/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=native_cpu,spir64 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-LIB-LOC-MIXED %s
// CHECK-LIB-LOC-MIXED: clang-linker-wrapper
// CHECK-LIB-LOC-MIXED-SAME: "--bitcode-library=native_cpu={{.*}}SYCL{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}dpcpp-7{{[/\\]+}}sycl{{[/\\]+}}libsycl-nativecpu_utils.bc"
// CHECK-LIB-LOC-MIXED-SAME: "-sycl-device-library-location={{.*}}SYCL{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}dpcpp-7{{[/\\]+}}sycl"

/// Check that a missing NativeCPU device library is diagnosed.
// RUN: not %clang -### -fsycl --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -ccc-install-dir %t.nonexistent/bin \
// RUN:   -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked \
// RUN:   -fsycl-targets=native_cpu \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-NATIVECPU-MISSING %s
// CHECK-NATIVECPU-MISSING: cannot find expected SYCL device library 'libsycl-nativecpu_utils.bc'

/// When the library cannot be found, fall back to <driver>/../lib.
// RUN: %clang -### -fsycl --offload-new-driver --target=x86_64-unknown-linux-gnu \
// RUN:   -ccc-install-dir %t.nonexistent/bin \
// RUN:   --no-offloadlib -fno-sycl-instrument-device-code \
// RUN:   -fsycl-targets=spir64 \
// RUN:   %s 2>&1 | FileCheck -check-prefix=CHECK-LIB-LOC-FALLBACK %s
// CHECK-LIB-LOC-FALLBACK: clang-linker-wrapper{{.*}} "-sycl-device-library-location={{.*}}.nonexistent{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib"
