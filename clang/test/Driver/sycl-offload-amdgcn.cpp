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
// CHK-ACTIONS: "-cc1" "-triple" "amdgcn-amd-amdhsa" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-target-cpu" "gfx906"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-compile-opts=--offload-arch=gfx906" "-target=amdgcn" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx906 %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NO-CC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 3: input, "{{.*}}", c++, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 5: compiler, {4}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (amdgcn-amd-amdhsa:gfx906)" {5}, c++-cpp-output
// CHK-PHASES-NO-CC: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-NO-CC: 10: linker, {5}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 11: sycl-post-link, {10}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 12: file-table-tform, {11}, ir, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 13: backend, {12}, assembler, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 14: assembler, {13}, object, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 15: linker, {14}, image, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 16: linker, {15}, hip-fatbin, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 17: foreach, {12, 16}, hip-fatbin, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 18: file-table-tform, {11, 17}, tempfiletable, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 19: clang-offload-wrapper, {18}, object, (device-sycl, gfx906)
// CHK-PHASES-NO-CC: 20: offload, "device-sycl (amdgcn-amd-amdhsa:gfx906)" {19}, object
// CHK-PHASES-NO-CC: 21: linker, {9, 20}, image, (host-sycl)

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
