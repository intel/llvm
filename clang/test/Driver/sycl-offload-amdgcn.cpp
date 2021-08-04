/// Tests specific to `-fsycl-targets=amdgcn-amd-amdhsa-sycldevice`
// REQUIRES: clang-driver

// UNSUPPORTED: system-windows

// Check that the offload arch is required
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ARCH %s
// CHK-ARCH: error: Missing AMDGPU architecture for SYCL offloading. Please specify it with -Xsycl-target-backend --offload-arch.

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -Xsycl-target-backend --offload-arch=gfx906 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
// CHK-ACTIONS: "-cc1" "-triple" "amdgcn-amd-amdhsa-sycldevice" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-target-cpu" "gfx906"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-compile-opts=--offload-arch=gfx906" "-target=amdgcn" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -Xsycl-target-backend --offload-arch=gfx906 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NO-CC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 3: input, "{{.*}}", c++, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 5: compiler, {4}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (amdgcn-amd-amdhsa-sycldevice:gfx906)" {5}, c++-cpp-output
// CHK-PHASES-NO-CC: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-NO-CC: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES-NO-CC: 11: linker, {5}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 12: sycl-post-link, {11}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 13: file-table-tform, {12}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 14: backend, {13}, assembler, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 15: assembler, {14}, object, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 16: linker, {15}, image, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 17: linker, {16}, hip-fatbin, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 18: foreach, {13, 17}, hip-fatbin, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 19: file-table-tform, {12, 18}, tempfiletable, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 20: clang-offload-wrapper, {19}, object, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (amdgcn-amd-amdhsa-sycldevice:gfx906)" {20}, image
