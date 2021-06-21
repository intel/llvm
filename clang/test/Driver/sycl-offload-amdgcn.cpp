/// Tests specific to `-fsycl-targets=amdgcn-amd-amdhsa-sycldevice`
// REQUIRES: clang-driver

// UNSUPPORTED: system-windows

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx906 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
// CHK-ACTIONS: "-cc1" "-triple" "amdgcn-amd-amdhsa-sycldevice" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-feature" "+ptx42"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=amdgcn" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl -fsycl-use-footer \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx906 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "vector_add.cpp", c++, (host-sycl)
// CHK-PHASES-NO-CC: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC:- 2: input, "vector_add.cpp", c++, (device-sycl)
// CHK-PHASES-NO-CC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl)
// CHK-PHASES-NO-CC: compiler, {3}, ir, (device-sycl)
// CHK-PHASES-NO-CC: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (amdgcn-amd-amdhsa-sycldevice)" {4}, c++-cpp-output
// CHK-PHASES-NO-CC: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-NO-CC: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: assembler, {7}, object, (host-sycl)
// CHK-PHASES-NO-CC: linker, {8}, image, (host-sycl)
// CHK-PHASES-NO-CC: linker, {4}, ir, (device-sycl)
// CHK-PHASES-NO-CC: sycl-post-link, {10}, ir, (device-sycl)
// CHK-PHASES-NO-CC: backend, {11}, assembler, (device-sycl)
// CHK-PHASES-NO-CC: assembler, {12}, object, (device-sycl)
// CHK-PHASES-NO-CC: linker, {13}, image, (device-sycl)
// CHK-PHASES-NO-CC:linker, {14}, hip-fatbin, (device-sycl)
// CHK-PHASES-NO-CC: clang-offload-wrapper, {15}, object, (device-sycl)
// CHK-PHASES-NO-CC: offload, "host-sycl (x86_64-unknown-linux-gnu)" {9}, "device-sycl (amdgcn-amd-amdhsa-sycldevice)" {16}, image
