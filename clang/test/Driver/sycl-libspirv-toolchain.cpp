// Test the search logic for the libspirv bitcode library in the offloading toolchains that need it.

// DEFINE: %{resource_dir} = %/S/Inputs/SYCL/lib/clang/resource_dir

// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-WINDOWS
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-gnu %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-WINDOWS
// CHECK-WINDOWS: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}remangled-l32-signed_char.libspirv-nvptx64-nvidia-cuda.bc"
//
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-linux-gnu %s 2>&1 \
// RUN: | FileCheck %s -DRESOURCE_DIR=%{resource_dir} --check-prefixes=CHECK-LINUX
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-cygnus %s 2>&1 \
// RUN: | FileCheck %s -DRESOURCE_DIR=%{resource_dir} --check-prefixes=CHECK-LINUX
// CHECK-LINUX: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[RESOURCE_DIR]]{{.*[\\/]}}remangled-l64-signed_char.libspirv-nvptx64-nvidia-cuda.bc"
//
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN: | FileCheck %s -DRESOURCE_DIR=%{resource_dir} --check-prefixes=CHECK-AMDGCN-WINDOWS
// CHECK-AMDGCN-WINDOWS: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[RESOURCE_DIR]]{{.*[\\/]}}remangled-l32-signed_char.libspirv-amdgcn-amd-amdhsa.bc"
//
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DRESOURCE_DIR=%{resource_dir} --check-prefixes=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[RESOURCE_DIR]]{{.*[\\/]}}remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc"
//
// Only link libspirv in SYCL language mode, `-fno-sycl-libspirv` should result in a warning
// RUN: %clang -### -x cu -fno-sycl-libspirv -nocudainc -nocudalib %s 2>&1 | FileCheck %s --check-prefixes=CHECK-CUDA
// CHECK-CUDA: warning: argument unused during compilation: '-fno-sycl-libspirv' [-Wunused-command-line-argument]
// CHECK-CUDA: "-cc1"{{.*}} "-fcuda-is-device"
// CHECK-CUDA-NOT: "-mlink-builtin-bitcode" "{{.*}}.libspirv-{{.*}}.bc"
//
// The path to the remangled libspirv bitcode file is determined by the resource directory
// RUN: %clang -### -resource-dir %{resource_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DRESOURCE_DIR=%{resource_dir} --check-prefixes=CHECK-DIR
// CHECK-DIR: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[RESOURCE_DIR]]{{.*[\\/]}}remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc"
//
// If libspirv path doesn't exist, error is reported.
// DEFINE: %{nonexistent_dir} = %/S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir
// RUN: not %clang -### -resource-dir %{nonexistent_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DDIR=%{nonexistent_dir} --check-prefixes=CHECK-HHH-NONEXISTENT
// CHECK-HHH-NONEXISTENT: error: cannot find 'remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
//
// RUN: not %clang -### -resource-dir %{nonexistent_dir} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s -DDIR=%{nonexistent_dir} --check-prefixes=CHECK-AMDGCN-HHH-NONEXISTENT
// CHECK-AMDGCN-HHH-NONEXISTENT: clang: error: cannot find 'remangled-{{.*}}.libspirv-amdgcn-amd-amdhsa.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
//
// RUN: not %clang -fdriver-only -resource-dir %{nonexistent_dir} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DDIR=%{nonexistent_dir} --check-prefixes=CHECK-DO-NONEXISTENT
// CHECK-DO-NONEXISTENT: error: cannot find 'remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
