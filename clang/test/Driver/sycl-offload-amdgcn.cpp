/// Tests specific to `-fsycl-targets=amdgcn-amd-amdhsa`

// UNSUPPORTED: system-windows
// REQUIRES: amdgpu-registered-target

// Check that the offload arch is required
// RUN: not %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ARCH %s
// CHK-ARCH: error: missing AMDGPU architecture for SYCL offloading; specify it with '-Xsycl-target-backend --offload-arch=<arch-name>'

// RUN: not %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=spir64,amdgcn-amd-amdhsa %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-MULTI-ARCH %s
// CHK-MULTI-ARCH: error: missing AMDGPU architecture for SYCL offloading; specify it with '-Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=<arch-name>'

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 -nogpulib\
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
// CHK-ACTIONS: "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-target-cpu" "gfx906"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-compile-opts=--offload-arch=gfx906" "-target=amdgcn" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -fsycl-device-lib=all -Xsycl-target-backend --offload-arch=gfx906 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 2: input, "{{.*}}", c++, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 4: compiler, {3}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (amdgcn-amd-amdhsa:gfx906)" {4}, c++-cpp-output
// CHK-PHASES-NO-CC: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-NO-CC: 9: linker, {4}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 10: input, "{{.*}}devicelib--amd.bc", ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 11: linker, {9, 10}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 12: sycl-post-link, {11}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 13: file-table-tform, {12}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 14: backend, {13}, assembler, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 15: assembler, {14}, object, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 16: linker, {15}, image, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 17: linker, {16}, hip-fatbin, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 18: foreach, {13, 17}, hip-fatbin, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 19: file-table-tform, {12, 18}, tempfiletable, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 20: clang-offload-wrapper, {19}, object, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 21: offload, "device-sycl (amdgcn-amd-amdhsa:gfx906)" {20}, object
// CHK-PHASES-NO-CC: 22: linker, {8, 21}, image, (host-sycl)

/// Check that we only unbundle an archive once.
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -nogpulib \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend \
// RUN:   --offload-arch=gfx906 %s -L%S/Inputs/SYCL -llin64 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ARCHIVE %s
// RUN: %clangxx -### -target x86_64-unknown-linux-gnu -fsycl -nogpulib \
// RUN:   -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend \
// RUN:   --offload-arch=gfx906 %s %S/Inputs/SYCL/liblin64.a 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ARCHIVE %s
// CHK-ARCHIVE: clang-offload-bundler{{.*}} "-input={{.*}}/Inputs/SYCL/liblin64.a"
// CHK-ARCHIVE: llvm-link{{.*}}
// CHK-ARCHIVE-NOT: clang-offload-bundler{{.*}} "-unbundle"
