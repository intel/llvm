/// In the new offloading model, -fsycl-targets cannot be used together with
/// --offload-arch when --offload-arch names an Intel architecture.

// DEFINE: %{clang_install_dir} = %/S/Inputs/SYCL/bin

// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=intel_gpu_pvc --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=GPU %s
// GPU: error: invalid argument '--offload-arch=pvc' not allowed with '-fsycl-targets=intel_gpu_pvc'

// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   --offload-arch=bmg_g21 -fsycl-targets=spir64,intel_gpu_pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ORDER %s
// ORDER: error: invalid argument '--offload-arch=bmg_g21' not allowed with '-fsycl-targets=spir64,intel_gpu_pvc'

// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64_x86_64 --offload-arch=skylake-avx512 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CPU %s
// CPU: error: invalid argument '--offload-arch=skylake-avx512' not allowed with '-fsycl-targets=spir64_x86_64'

/// --offload-arch passed with -Xarch_device or -Xarch_<arch> is diagnosed too.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_device --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=XARCH-DEVICE %s
// XARCH-DEVICE: error: invalid argument '-Xarch_device --offload-arch=pvc' not allowed with '-fsycl-targets=spir64'
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_spir64 --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=XARCH %s
// XARCH: error: invalid argument '-Xarch_spir64 --offload-arch=pvc' not allowed with '-fsycl-targets=spir64'

/// spir64_gen is not supported with the new offloading model, which is the
/// only error reported.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64_gen --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SPIR64-GEN %s
// SPIR64-GEN-NOT: not allowed with
// SPIR64-GEN: error: '-fsycl-targets=spir64_gen' is not supported with the new offloading model
// SPIR64-GEN-NOT: not allowed with

/// --offload-arch with a non-Intel architecture, and --offload-targets with
/// --offload-arch, are not affected.
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fno-sycl-libspirv -nocudalib -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --offload-arch=sm_80 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,NVPTX %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fno-sycl-libspirv -nocudalib -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --cuda-gpu-arch=sm_80 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,NVPTX %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fno-sycl-libspirv -nocudalib -fsycl-targets=amdgcn-amd-amdhsa \
// RUN:   --offload-arch=gfx908 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,AMDGCN %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   --offload-targets=spirv64-unknown-unknown --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,SPIRV %s
// NO-ERROR-NOT: error:
// NVPTX: "-triple" "nvptx64-nvidia-cuda"{{.*}} "-fsycl-is-device"
// AMDGCN: "-triple" "amdgpu{{.*}}-amd-amdhsa"{{.*}} "-fsycl-is-device"
// SPIRV: "-triple" "spirv64-unknown-unknown"{{.*}} "-fsycl-is-device"

/// The old offloading model does not support --offload-arch for SYCL, so the
/// options are not diagnosed as conflicting there.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=intel_gpu_pvc --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OLD-MODEL %s
// OLD-MODEL-NOT: not allowed with
// OLD-MODEL: "-triple" "spir64_gen-unknown-unknown"{{.*}} "-fsycl-is-device"
