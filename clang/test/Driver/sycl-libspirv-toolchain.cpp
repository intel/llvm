// Test the search logic for the libspirv bitcode library in the offloading toolchains that need it.

// DEFINE: %{resource_dir} = %/S/Inputs/SYCL/lib/clang/resource_dir

// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-WINDOWS
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-gnu %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-WINDOWS
// CHECK-WINDOWS: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}libspirv.l32.signed_char.bc"
//
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-linux-gnu %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-LINUX
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-cygnus %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-LINUX
// CHECK-LINUX: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}libspirv.l64.signed_char.bc"
//
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-AMDGCN-WINDOWS
// CHECK-AMDGCN-WINDOWS: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}libspirv.l32.signed_char.bc"
//
// DEFINE: %{amdgcn} = %clang -### -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib -target x86_64-unknown-linux-gnu %s
// DEFINE: %{amd_gpu} = %clang -### -fsycl -fsycl-device-only -fsycl-targets=amd_gpu_gfx908 -nogpulib -target x86_64-unknown-linux-gnu %s
// RUN: rm -rf %t
// RUN: mkdir -p %t/amdgpu/lib/amdgpu-amd-amdhsa %t/amdgpu-llvm/lib/amdgpu-amd-amdhsa-llvm %t/amdgcn/lib/amdgcn-amd-amdhsa %t/amdgcn-llvm/lib/amdgcn-amd-amdhsa-llvm
// RUN: touch %t/amdgpu/lib/amdgpu-amd-amdhsa/libspirv.l64.signed_char.bc
// RUN: touch %t/amdgpu-llvm/lib/amdgpu-amd-amdhsa-llvm/libspirv.l64.signed_char.bc
// RUN: touch %t/amdgcn/lib/amdgcn-amd-amdhsa/libspirv.l64.signed_char.bc
// RUN: touch %t/amdgcn-llvm/lib/amdgcn-amd-amdhsa-llvm/libspirv.l64.signed_char.bc
// RUN: %{amdgcn} -resource-dir %t/amdgpu 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu-amd-amdhsa
// RUN: %{amd_gpu} -resource-dir %t/amdgpu 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu-amd-amdhsa
// RUN: %{amdgcn} -resource-dir %t/amdgpu-llvm 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu-amd-amdhsa-llvm
// RUN: %{amd_gpu} -resource-dir %t/amdgpu-llvm 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu-amd-amdhsa-llvm
// RUN: %{amdgcn} -resource-dir %t/amdgcn 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgcn-amd-amdhsa
// RUN: %{amd_gpu} -resource-dir %t/amdgcn 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgcn-amd-amdhsa
// RUN: %{amdgcn} -resource-dir %t/amdgcn-llvm 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgcn-amd-amdhsa-llvm
// RUN: %{amd_gpu} -resource-dir %t/amdgcn-llvm 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgcn-amd-amdhsa-llvm
// RUN: cp -R %t/amdgpu-llvm/lib/amdgpu-amd-amdhsa-llvm %t/amdgcn/lib/amdgcn-amd-amdhsa %t/amdgcn-llvm/lib/amdgcn-amd-amdhsa-llvm %t/amdgpu/lib
// RUN: %{amdgcn} -resource-dir %t/amdgpu 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu-amd-amdhsa
// RUN: %{amd_gpu} -resource-dir %t/amdgpu 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu-amd-amdhsa
// RUN: mkdir -p %t/amdgpu/lib/amdgpu9.08-amd-amdhsa
// RUN: touch %t/amdgpu/lib/amdgpu9.08-amd-amdhsa/libspirv.l64.signed_char.bc
// RUN: %{amdgcn} -resource-dir %t/amdgpu 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu9.08-amd-amdhsa
// RUN: %{amd_gpu} -resource-dir %t/amdgpu 2>&1 | FileCheck %s --check-prefix=CHECK-AMD-FALLBACK -DTRIPLE=amdgpu9.08-amd-amdhsa
// CHECK-AMD-FALLBACK: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]+}}lib{{[\\/]+}}[[TRIPLE]]{{[\\/]+}}libspirv.l64.signed_char.bc"
//
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}libspirv.l[[#]].signed_char.bc"
//
// Only link libspirv in SYCL language mode, `-fno-sycl-libspirv` should result in a warning
// RUN: %clang -### -x cu -fno-sycl-libspirv -nocudainc -nocudalib %s 2>&1 | FileCheck %s --check-prefixes=CHECK-CUDA
// CHECK-CUDA: warning: argument unused during compilation: '-fno-sycl-libspirv' [-Wunused-command-line-argument]
// CHECK-CUDA: "-cc1"{{.*}} "-fcuda-is-device"
// CHECK-CUDA-NOT: "-mlink-builtin-bitcode" "{{.*}}libspirv.l[[#]].signed_char.bc"
//
// The path to the libspirv bitcode file is determined by the resource directory
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DRESOURCE_DIR=%{resource_dir} --check-prefixes=CHECK-DIR
// CHECK-DIR: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[RESOURCE_DIR]]{{.*[\\/]}}libspirv.l[[#]].signed_char.bc"
//
// If libspirv path doesn't exist, error is reported.
// DEFINE: %{nonexistent_dir} = %/S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir
// RUN: not %clang -### -resource-dir %{nonexistent_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DDIR=%{nonexistent_dir} --check-prefixes=CHECK-HHH-NONEXISTENT
// CHECK-HHH-NONEXISTENT: error: cannot find 'libspirv.l[[#]].signed_char.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
//
// RUN: not %clang -### -resource-dir %{nonexistent_dir} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s -DDIR=%{nonexistent_dir} --check-prefixes=CHECK-AMDGCN-HHH-NONEXISTENT
// CHECK-AMDGCN-HHH-NONEXISTENT: clang: error: cannot find 'libspirv.l[[#]].signed_char.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
//
// RUN: not %clang -fdriver-only -resource-dir %{nonexistent_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DDIR=%{nonexistent_dir} --check-prefixes=CHECK-DO-NONEXISTENT
// CHECK-DO-NONEXISTENT: error: cannot find 'libspirv.l[[#]].signed_char.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
