/// Tests specific to `-fsycl-targets=amdgcn-amd-amdhsa-sycldevice`
// REQUIRES: clang-driver

// UNSUPPORTED: system-windows

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx906 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
// CHK-ACTIONS: "-cc1" "-triple" "amdgcn-amd-amdhsa-sycldevice" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-target-cpu" "gfx906"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=amdgcn" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -mcpu=gfx906 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NO-CC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 3: input, "{{.*}}", c++, (device-sycl)
// CHK-PHASES-NO-CC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHK-PHASES-NO-CC: 5: compiler, {4}, ir, (device-sycl)
// CHK-PHASES-NO-CC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (amdgcn-amd-amdhsa-sycldevice)" {5}, c++-cpp-output
// CHK-PHASES-NO-CC: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-NO-CC: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES-NO-CC: 11: linker, {5}, ir, (device-sycl)
// CHK-PHASES-NO-CC: 12: sycl-post-link, {11}, ir, (device-sycl)
// CHK-PHASES-NO-CC: 13: backend, {12}, assembler, (device-sycl)
// CHK-PHASES-NO-CC: 14: assembler, {13}, object, (device-sycl)
// CHK-PHASES-NO-CC: 15: linker, {14}, image, (device-sycl)
// CHK-PHASES-NO-CC: 16: linker, {15}, hip-fatbin, (device-sycl)
// CHK-PHASES-NO-CC: 17: clang-offload-wrapper, {16}, object, (device-sycl)
// CHK-PHASES-NO-CC: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (amdgcn-amd-amdhsa-sycldevice)" {17}, image
