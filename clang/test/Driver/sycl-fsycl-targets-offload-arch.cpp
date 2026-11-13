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

/// The option is reported as the user spelled it.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 --cuda-gpu-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ALIAS %s
// ALIAS: error: invalid argument '--cuda-gpu-arch=pvc' not allowed with '-fsycl-targets=spir64'

/// Several Intel architectures are reported once.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 --offload-arch=pvc --offload-arch=bmg_g21 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=MULTI %s
// MULTI: error: invalid argument '--offload-arch=pvc' not allowed with '-fsycl-targets=spir64'
// MULTI-NOT: not allowed with

/// This also applies to -fsycl-device-only and to -fsycl-targets=native_cpu.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 --offload-arch=pvc -fsycl-device-only -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SPIR64 %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=native_cpu --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=NATIVE-CPU %s
// SPIR64: error: invalid argument '--offload-arch=pvc' not allowed with '-fsycl-targets=spir64'
// NATIVE-CPU: error: invalid argument '--offload-arch=pvc' not allowed with '-fsycl-targets=native_cpu'

/// An architecture removed with --no-offload-arch, or passed to the host
/// compilation, is not diagnosed.
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 --offload-arch=pvc --no-offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,SPIRV64-JIT %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   --offload-arch=pvc --no-offload-arch=pvc -fsycl-targets=intel_gpu_pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,SPIR64-GEN-PVC %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   --offload-arch=pvc --no-offload-arch=all -fsycl-targets=intel_gpu_pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,SPIR64-GEN-PVC %s
// SPIR64-GEN-PVC: "-triple" "spir64_gen-unknown-unknown"{{.*}} "-fsycl-is-device"
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_host --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,SPIRV64-JIT %s
// SPIRV64-JIT: "-triple" "spir64-unknown-unknown"{{.*}} "-fsycl-is-device"

/// --offload-arch passed to the SYCL device compilation with -Xarch_device,
/// -Xarch_<arch> or -Xsycl-target-backend is diagnosed too, once.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_device --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FORWARDED %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_device "-O2 --offload-arch=pvc" -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FORWARDED %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_spir64 --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FORWARDED %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xsycl-target-backend --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FORWARDED %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xsycl-target-backend=spir64 --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=FORWARDED %s
// FORWARDED: error: invalid argument '--offload-arch=pvc' not allowed with '-fsycl-targets=spir64'
// FORWARDED-NOT: not allowed with

/// --offload-arch for another device is not diagnosed: -Xarch_spir64 when
/// there is no SPIR-V device, and -Xsycl-target-backend for an NVIDIA device.
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fno-sycl-libspirv -nocudalib -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:   --offload-arch=sm_80 -Xarch_spir64 --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,NVPTX %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fno-sycl-libspirv -nocudalib -fsycl-targets=spir64,nvptx64-nvidia-cuda \
// RUN:   -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_80 -c %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,NVPTX %s

/// spir64_gen is not supported with the new offloading model, which is the
/// only error reported.
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64_gen --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SPIR64-GEN %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -fsycl-targets=spir64_gen --offload-arch=pvc -c %s 2>&1 \
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

/// The options are also diagnosed when only linking.
// RUN: touch %t.o
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 --offload-arch=pvc %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK %s
// RUN: not %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 -Xarch_device --offload-arch=pvc %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=LINK %s
// LINK: error: invalid argument '--offload-arch=pvc' not allowed with '-fsycl-targets=spir64'
// LINK-NOT: not allowed with
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=spir64 %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,LINK-OK %s
// RUN: %clangxx -### -fsycl --offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   --offload-arch=pvc %t.o 2>&1 \
// RUN:   | FileCheck -check-prefixes=NO-ERROR,LINK-OK %s
// LINK-OK: clang-linker-wrapper

/// The old offloading model does not support --offload-arch for SYCL, so the
/// options are not diagnosed as conflicting there.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -ccc-install-dir %{clang_install_dir} \
// RUN:   -fsycl-targets=intel_gpu_pvc --offload-arch=pvc -c %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OLD-MODEL %s
// OLD-MODEL-NOT: not allowed with
// OLD-MODEL: "-triple" "spir64_gen-unknown-unknown"{{.*}} "-fsycl-is-device"
