/// Tests specific to `-fsycl-targets=nvptx64-nvidia-nvcl-sycldevice`
// REQUIRES: clang-driver

// UNSUPPORTED: system-windows

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s

// CHK-ACTIONS: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-emit-llvm-bc" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-feature" "+ptx42"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS: sycl-post-link{{.*}} "-split=auto"
// CHK-ACTIONS: file-table-tform" "-extract=Code" "-drop_titles"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}clang-{{[0-9]+}}"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}ptxas"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}fatbinary"
// CHK-ACTIONS: file-table-tform" "-replace=Code,Code"
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=nvptx64" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NO-CC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 3: input, "{{.*}}", c++, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 5: compiler, {4}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {5}, c++-cpp-output
// CHK-PHASES-NO-CC: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-NO-CC: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES-NO-CC: 11: linker, {5}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 12: sycl-post-link, {11}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 13: file-table-tform, {12}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 14: backend, {13}, assembler, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 15: assembler, {14}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 16: linker, {14, 15}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 17: foreach, {13, 16}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 18: file-table-tform, {12, 17}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 19: clang-offload-wrapper, {18}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 20: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {19}, image

/// Check phases specifying a compute capability.
// RUN: %clangxx -ccc-print-phases -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda \
N.)
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES %s
// CHK-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 3: input, "{{.*}}", c++, (device-sycl, sm_35)
// CHK-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES: 5: compiler, {4}, ir, (device-sycl, sm_35)
// CHK-PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {5}, c++-cpp-output
// CHK-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES: 11: linker, {5}, ir, (device-sycl, sm_35)
// CHK-PHASES: 12: sycl-post-link, {11}, ir, (device-sycl, sm_35)
// CHK-PHASES: 13: file-table-tform, {12}, ir, (device-sycl, sm_35)
// CHK-PHASES: 14: backend, {13}, assembler, (device-sycl, sm_35)
// CHK-PHASES: 15: assembler, {14}, object, (device-sycl, sm_35)
// CHK-PHASES: 16: linker, {14, 15}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 17: foreach, {13, 16}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 18: file-table-tform, {12, 17}, tempfiletable, (device-sycl, sm_35)
// CHK-PHASES: 19: clang-offload-wrapper, {18}, object, (device-sycl, sm_35)
// CHK-PHASES: 20: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {19}, image
