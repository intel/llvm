// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-WINDOWS
// CHECK-WINDOWS: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}remangled-l32-signed_char.libspirv-nvptx64-nvidia-cuda.bc"
//
// RUN: %clang -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib -target x86_64-unknown-linux-gnu %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-LINUX
// CHECK-LINUX: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}remangled-l64-signed_char.libspirv-nvptx64-nvidia-cuda.bc"
//
// AMDGCN wrongly uses 32-bit longs on Windows
// RUN: %clang -### -resource-dir %S/Inputs/SYCL/lib/clang/resource_dir -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib -target x86_64-unknown-windows-msvc %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-AMDGCN-WINDOWS
// CHECK-AMDGCN-WINDOWS: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "{{.*[\\/]}}remangled-l64-signed_char.libspirv-amdgcn-amd-amdhsa.bc"
//
// RUN: %clang -### -fsycl -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "-cc1"{{.*}} "-fsycl-is-device"
// CHECK-DEVICE-ONLY-NOT: "-mlink-builtin-bitcode" "{{.*}}.libspirv-{{.*}}.bc"
//
// Only link libspirv in SYCL language mode, but `-fno-sycl-libspirv` does not result in a warning
// RUN: %clang -### -x cu -fno-sycl-libspirv -nocudainc -nocudalib %s 2>&1 | FileCheck %s --check-prefixes=CHECK-CUDA
// CHECK-CUDA-NOT: warning: argument unused during compilation: '-fno-sycl-libspirv' [-Wunused-command-line-argument]
// CHECK-CUDA: "-cc1"{{.*}} "-fcuda-is-device"
// CHECK-CUDA-NOT: "-mlink-builtin-bitcode" "{{.*}}.libspirv-{{.*}}.bc"
//
// The path to the remangled libspirv bitcode file is determined by the resource directory.
// RUN: %clang -### -ccc-install-dir %S/Inputs/SYCL/bin -resource-dir %S/Inputs/SYCL/lib/clang/resource_dir -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s -DINSTALL_DIR=%S/Inputs/SYCL/bin -DRESOURCE_DIR=%S/Inputs/SYCL/lib/clang/resource_dir --check-prefixes=CHECK-DIR
// CHECK-DIR: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[RESOURCE_DIR]]{{.*[\\/]}}remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc"
//
// The `-###` option disables file existence checks
// RUN: %clang -### -resource-dir %S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s      -DDIR=%S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir --check-prefixes=CHECK-HHH-NOT-FOUND
// CHECK-HHH-NOT-FOUND: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-mlink-builtin-bitcode" "[[DIR]]{{.*[\\/]}}remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc"
//
// But not for AMDGCN :^)
// RUN: not %clang -### -resource-dir %S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx908 -nogpulib %s 2>&1 \
// RUN: | FileCheck %s      -DDIR=%S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir --check-prefixes=CHECK-AMDGCN-HHH-NOT-FOUND
// CHECK-AMDGCN-HHH-NOT-FOUND: error: cannot find 'remangled-{{.*}}.libspirv-amdgcn-amd-amdhsa.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
//
// `-fdriver-only` has no such special handling, so it will not find the file
// RUN: not %clang -fdriver-only -resource-dir %S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir -fsycl -fsycl-targets=nvptx64-nvidia-cuda -nocudalib %s 2>&1 \
// RUN: | FileCheck %s              -DDIR=%S/Inputs/SYCL/does_not_exist/lib/clang/resource_dir --check-prefixes=CHECK-DO-NOT-FOUND
// CHECK-DO-NOT-FOUND: error: cannot find 'remangled-{{.*}}.libspirv-nvptx64-nvidia-cuda.bc'; provide path to libspirv library via '-fsycl-libspirv-path', or pass '-fno-sycl-libspirv' to build without linking with libspirv
