// REQUIRES: system-linux
/// Verify --offload-new-driver option phases
// RUN:  %clang --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64 --offload-new-driver -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix=OFFLOAD-NEW-DRIVER %s
// OFFLOAD-NEW-DRIVER: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// OFFLOAD-NEW-DRIVER: 1: append-footer, {0}, c++, (host-sycl)
// OFFLOAD-NEW-DRIVER: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// OFFLOAD-NEW-DRIVER: 3: compiler, {2}, ir, (host-sycl)
// OFFLOAD-NEW-DRIVER: 4: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// OFFLOAD-NEW-DRIVER: 5: preprocessor, {4}, c++-cpp-output, (device-sycl, sm_50)
// OFFLOAD-NEW-DRIVER: 6: compiler, {5}, ir, (device-sycl, sm_50)
// OFFLOAD-NEW-DRIVER: 7: backend, {6}, ir, (device-sycl, sm_50)
// OFFLOAD-NEW-DRIVER: 8: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {7}, ir
// OFFLOAD-NEW-DRIVER: 9: input, "[[INPUT]]", c++, (device-sycl)
// OFFLOAD-NEW-DRIVER: 10: preprocessor, {9}, c++-cpp-output, (device-sycl)
// OFFLOAD-NEW-DRIVER: 11: compiler, {10}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 12: backend, {11}, ir, (device-sycl)
// OFFLOAD-NEW-DRIVER: 13: offload, "device-sycl (spir64-unknown-unknown)" {12}, ir
// OFFLOAD-NEW-DRIVER: 14: clang-offload-packager, {8, 13}, image, (device-sycl)
// OFFLOAD-NEW-DRIVER: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {14}, ir
// OFFLOAD-NEW-DRIVER: 16: backend, {15}, assembler, (host-sycl)
// OFFLOAD-NEW-DRIVER: 17: assembler, {16}, object, (host-sycl)
// OFFLOAD-NEW-DRIVER: 18: clang-linker-wrapper, {17}, image, (host-sycl)

/// Check the toolflow for SYCL compilation using new offload model
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl -fsycl-targets=spir64 --offload-new-driver %s 2>&1 | FileCheck -check-prefix=CHK-FLOW %s
// CHK-FLOW: clang{{.*}} "-cc1" "-triple" "spir64-unknown-unknown" "-aux-triple" "x86_64-unknown-linux-gnu" "-fsycl-is-device" {{.*}} "-fsycl-int-header=[[HEADER:.*]].h" "-fsycl-int-footer=[[FOOTER:.*]].h" {{.*}} "--offload-new-driver" {{.*}} "-o" "[[CC1DEVOUT:.*]]" "-x" "c++" "[[INPUT:.*]]"
// CHK-FLOW-NEXT: clang-offload-packager{{.*}} "-o" "[[PACKOUT:.*]]" "--image=file=[[CC1DEVOUT]],triple=spir64-unknown-unknown,arch=,kind=sycl{{.*}}"
// CHK-FLOW-NEXT: append-file{{.*}} "[[INPUT]]" "--append=[[FOOTER]].h" "--orig-filename=[[INPUT]]" "--output=[[APPENDOUT:.*]]" "--use-include"
// CHK-FLOW-NEXT: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-include" "[[HEADER]].h" "-dependency-filter" "[[HEADER]].h" {{.*}} "-fsycl-is-host"{{.*}} "-full-main-file-name" "[[INPUT]]" {{.*}} "--offload-new-driver" {{.*}} "-fembed-offload-object=[[PACKOUT]]" {{.*}} "-o" "[[CC1FINALOUT:.*]]" "-x" "c++" "[[APPENDOUT]]"
// CHK-FLOW-NEXT: clang-linker-wrapper{{.*}} "--host-triple=x86_64-unknown-linux-gnu"{{.*}} "--linker-path={{.*}}/ld" {{.*}} "[[CC1FINALOUT]]"

/// Verify options passed to clang-linker-wrapper
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          --sysroot=%S/Inputs/SYCL -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS %s
// WRAPPER_OPTIONS: clang-linker-wrapper{{.*}} "-sycl-device-libraries=libsycl-crt.new.o,libsycl-complex.new.o,libsycl-complex-fp64.new.o,libsycl-cmath.new.o,libsycl-cmath-fp64.new.o,libsycl-imf.new.o,libsycl-imf-fp64.new.o,libsycl-imf-bf16.new.o,libsycl-fallback-cassert.new.o,libsycl-fallback-cstring.new.o,libsycl-fallback-complex.new.o,libsycl-fallback-complex-fp64.new.o,libsycl-fallback-cmath.new.o,libsycl-fallback-cmath-fp64.new.o,libsycl-fallback-imf.new.o,libsycl-fallback-imf-fp64.new.o,libsycl-fallback-imf-bf16.new.o,libsycl-itt-user-wrappers.new.o,libsycl-itt-compiler-wrappers.new.o,libsycl-itt-stubs.new.o"
// WRAPPER_OPTIONS-SAME: "-sycl-device-library-location={{.*}}/lib"

/// Verify phases used to generate SPIR-V instead of LLVM-IR
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-device-obj=spirv -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix SPIRV_OBJ %s
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-device-only -fsycl-device-obj=spirv \
// RUN:          -ccc-print-phases %s 2>&1 \
// RUN:   | FileCheck -check-prefix SPIRV_OBJ %s
// SPIRV_OBJ: [[#SPVOBJ:]]: input, "{{.*}}", c++, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+1]]: preprocessor, {[[#SPVOBJ]]}, c++-cpp-output, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+2]]: compiler, {[[#SPVOBJ+1]]}, ir, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+3]]: backend, {[[#SPVOBJ+2]]}, ir, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+4]]: llvm-spirv, {[[#SPVOBJ+3]]}, spirv, (device-sycl)
// SPIRV_OBJ: [[#SPVOBJ+5]]: offload, "device-sycl (spir64-unknown-unknown)" {[[#SPVOBJ+4]]}, {{.*}}

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xspirv-translator -translator-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_TRANSLATOR %s
// WRAPPER_OPTIONS_TRANSLATOR: clang-linker-wrapper{{.*}} "--llvm-spirv-options={{.*}}-translator-opt{{.*}}"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xdevice-post-link -post-link-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_POSTLINK %s
// WRAPPER_OPTIONS_POSTLINK: clang-linker-wrapper{{.*}} "--sycl-post-link-options=-O2 -device-globals -properties -post-link-opt"

// -fsycl-device-only behavior
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-device-only -ccc-print-phases %s 2>&1 \
// RUN   | FileCheck -check-prefix DEVICE_ONLY %s
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          --offload-device-only -ccc-print-phases %s 2>&1 \
// RUN:  | FileCheck -check-prefix DEVICE_ONLY %s
// DEVICE_ONLY: 0: input, "{{.*}}", c++, (device-sycl)
// DEVICE_ONLY: 1: preprocessor, {0}, c++-cpp-output, (device-sycl)
// DEVICE_ONLY: 2: compiler, {1}, ir, (device-sycl)
// DEVICE_ONLY: 3: backend, {2}, ir, (device-sycl)
// DEVICE_ONLY: 4: offload, "device-sycl (spir64-unknown-unknown)" {3}, none

/// check for -shared transmission to clang-linker-wrapper tool
// RUN: %clangxx -### -fsycl --offload-new-driver -target x86_64-unknown-linux-gnu \
// RUN:          -shared %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK_SHARED %s
// CHECK_SHARED: clang-linker-wrapper{{.*}} "-shared"

// Verify 'arch' offload-packager values for known targets
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fsycl-targets=spir64 --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_ARCH \
// RUN:              -DTRIPLE=spir64-unknown-unknown -DARCH= %s
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fsycl-targets=intel_gpu_pvc --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_ARCH \
// RUN:              -DTRIPLE=spir64_gen-unknown-unknown -DARCH=pvc %s
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \
// RUN:          "-device pvc" --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_ARCH \
// RUN:              -DTRIPLE=spir64_gen-unknown-unknown -DARCH=pvc %s
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \
// RUN:          "-device pvc" -Xsycl-target-backend=spir64_gen "-device dg1" \
// RUN:          --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_ARCH \
// RUN:              -DTRIPLE=spir64_gen-unknown-unknown -DARCH=dg1 %s
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fno-sycl-libspirv -fsycl-targets=amd_gpu_gfx900 \
// RUN:          -nogpulib --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_ARCH \
// RUN:              -DTRIPLE=amdgcn-amd-amdhsa -DARCH=gfx900 %s
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fno-sycl-libspirv -fsycl-targets=nvidia_gpu_sm_50 \
// RUN:          -nogpulib --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_ARCH \
// RUN:              -DTRIPLE=nvptx64-nvidia-cuda -DARCH=sm_50 %s
// CHK_ARCH: clang{{.*}} "-triple" "[[TRIPLE]]"
// CHK_ARCH-SAME: "-fsycl-is-device" {{.*}} "--offload-new-driver"{{.*}} "-o" "[[CC1DEVOUT:.+\.bc]]"
// CHK_ARCH-NEXT: clang-offload-packager{{.*}} "--image=file=[[CC1DEVOUT]],triple=[[TRIPLE]],arch=[[ARCH]],kind=sycl{{.*}}"

// Verify offload-packager option values
// RUN: %clangxx -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fsycl-targets=spir64,intel_gpu_pvc \
// RUN:          -Xsycl-target-backend=spir64 -spir64-opt \
// RUN:          -Xsycl-target-backend=intel_gpu_pvc -spir64_gen-opt \
// RUN:          -Xsycl-target-linker=spir64 -spir64-link-opt \
// RUN:          -Xsycl-target-linker=intel_gpu_pvc -spir64_gen-link-opt \
// RUN:          --offload-new-driver %s 2>&1 \
// RUN:  | FileCheck -check-prefix=CHK_PACKAGER_OPTS %s
// CHK_PACKAGER_OPTS: clang-offload-packager{{.*}} "-o"
// CHK_PACKAGER_OPTS-SAME: {{.*}}triple=spir64_gen-unknown-unknown,arch=pvc,kind=sycl,compile-opts={{.*}}-spir64_gen-opt,link-opts=-spir64_gen-link-opt
// CHK_PACKAGER_OPTS-SAME: {{.*}}triple=spir64-unknown-unknown,arch=,kind=sycl,compile-opts={{.*}}-spir64-opt,link-opts=-spir64-link-opt

/// Check phases with multiple intel_gpu settings
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:          -fsycl-targets=intel_gpu_dg1,intel_gpu_pvc \
// RUN:          --offload-new-driver -ccc-print-phases %s 2>&1 \
// RUN:  | FileCheck -check-prefix=MULT_TARG_PHASES %s
// MULT_TARG_PHASES: 0: input, "[[INPUT:.+\.c]]", c++, (host-sycl)
// MULT_TARG_PHASES: 1: append-footer, {0}, c++, (host-sycl)
// MULT_TARG_PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// MULT_TARG_PHASES: 3: compiler, {2}, ir, (host-sycl)
// MULT_TARG_PHASES: 4: input, "[[INPUT]]", c++, (device-sycl, dg1)
// MULT_TARG_PHASES: 5: preprocessor, {4}, c++-cpp-output, (device-sycl, dg1)
// MULT_TARG_PHASES: 6: compiler, {5}, ir, (device-sycl, dg1)
// MULT_TARG_PHASES: 7: backend, {6}, ir, (device-sycl, dg1)
// MULT_TARG_PHASES: 8: offload, "device-sycl (spir64_gen-unknown-unknown:dg1)" {7}, ir
// MULT_TARG_PHASES: 9: input, "[[INPUT]]", c++, (device-sycl, pvc)
// MULT_TARG_PHASES: 10: preprocessor, {9}, c++-cpp-output, (device-sycl, pvc)
// MULT_TARG_PHASES: 11: compiler, {10}, ir, (device-sycl, pvc)
// MULT_TARG_PHASES: 12: backend, {11}, ir, (device-sycl, pvc)
// MULT_TARG_PHASES: 13: offload, "device-sycl (spir64_gen-unknown-unknown:pvc)" {12}, ir
// MULT_TARG_PHASES: 14: clang-offload-packager, {8, 13}, image, (device-sycl)
// MULT_TARG_PHASES: 15: offload, "host-sycl (x86_64-unknown-linux-gnu)" {3}, "device-sycl (x86_64-unknown-linux-gnu)" {14}, ir
// MULT_TARG_PHASES: 16: backend, {15}, assembler, (host-sycl)
// MULT_TARG_PHASES: 17: assembler, {16}, object, (host-sycl)

/// Test option passing behavior for clang-offload-wrapper options.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xsycl-target-backend -backend-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_BACKEND %s
// WRAPPER_OPTIONS_BACKEND: clang-linker-wrapper{{.*}} "--sycl-backend-compile-options={{.*}}-backend-opt{{.*}}"

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -Xsycl-target-linker -link-opt -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_LINK %s
// WRAPPER_OPTIONS_LINK: clang-linker-wrapper{{.*}} "--sycl-target-link-options={{.*}}-link-opt{{.*}}"

/// Test option passing behavior for clang-offload-wrapper options for AOT.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:          -fsycl-targets=spir64_gen,spir64_x86_64 \
// RUN:          -Xsycl-target-backend=spir64_gen -backend-gpu-opt \
// RUN:          -Xsycl-target-backend=spir64_x86_64 -backend-cpu-opt \
// RUN:          -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix WRAPPER_OPTIONS_BACKEND_AOT %s
// WRAPPER_OPTIONS_BACKEND_AOT: clang-linker-wrapper{{.*}}  "--host-triple=x86_64-unknown-linux-gnu"
// WRAPPER_OPTIONS_BACKEND_AOT-SAME: "--gpu-tool-arg={{.*}}-backend-gpu-opt"
// WRAPPER_OPTIONS_BACKEND_AOT-SAME: "--cpu-tool-arg={{.*}}-backend-cpu-opt"

/// Verify arch settings for nvptx and amdgcn targets
// RUN: %clangxx -fsycl -### -fsycl-targets=amdgcn-amd-gpu -fno-sycl-libspirv \
// RUN:          -nocudalib --offload-new-driver \
// RUN:          -Xsycl-target-backend=amdgcn-amd-gpu --offload-arch=gfx600 \
// RUN:          %s 2>&1 \
// RUN:   | FileCheck -check-prefix AMD_ARCH %s
// AMD_ARCH: clang-offload-packager{{.*}} "--image=file={{.*}},triple=amdgcn-amd-gpu,arch=gfx600,kind=sycl,compile-opts=--offload-arch=gfx600"

// RUN: %clangxx -fsycl -### -fsycl-targets=nvptx64-nvidia-cuda \
// RUN:          -fno-sycl-libspirv -nocudalib --offload-new-driver %s 2>&1 \
// RUN:   | FileCheck -check-prefix NVPTX_DEF_ARCH %s
// NVPTX_DEF_ARCH: clang-offload-packager{{.*}} "--image=file={{.*}},triple=nvptx64-nvidia-cuda,arch=sm_50,kind=sycl"
