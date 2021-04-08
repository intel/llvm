/// Tests specific to `-fsycl-targets=nvptx64-nvidia-nvcl-sycldevice`
// REQUIRES: clang-driver

// UNSUPPORTED: system-windows

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
// CHK-ACTIONS: "-cc1" "-triple" "nvptx64-nvidia-nvcl-sycldevice"{{.*}} "-fsycl-is-device"{{.*}} "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-Wno-sycl-strict" "-sycl-std=2020" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-feature" "+ptx42"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=nvptx64" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 2: input, "{{.*}}", c++, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 4: compiler, {3}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_50)" {4}, c++-cpp-output
// CHK-PHASES-NO-CC: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-NO-CC: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES-NO-CC: 10: linker, {4}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 11: sycl-post-link, {10}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 12: backend, {11}, assembler, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 13: clang-offload-wrapper, {12}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 14: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_50)" {13}, image

/// Check phases specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-nvcl-sycldevice \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "{{.*}}", c++, (device-sycl, sm_35)
// CHK-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES: 4: compiler, {3}, ir, (device-sycl, sm_35)
// CHK-PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_35)" {4}, c++-cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {8}, image, (host-sycl)
// CHK-PHASES: 10: linker, {4}, ir, (device-sycl, sm_35)
// CHK-PHASES: 11: sycl-post-link, {10}, ir, (device-sycl, sm_35)
// CHK-PHASES: 12: backend, {11}, assembler, (device-sycl, sm_35)
// CHK-PHASES: 13: clang-offload-wrapper, {12}, object, (device-sycl, sm_35)
// CHK-PHASES: 14: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (nvptx64-nvidia-nvcl-sycldevice:sm_35)" {13}, image
