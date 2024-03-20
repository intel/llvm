/// Tests specific to `-fsycl-targets=nvptx64-nvidia-nvcl`

// UNSUPPORTED: system-windows

/// Check action graph.
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS %s
//
// RUN: %clang_cl -### -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS-WIN %s

// CHK-ACTIONS: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-emit-llvm-bc" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-target-feature" "+ptx42"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS: sycl-post-link{{.*}} "-split=auto"
// CHK-ACTIONS: file-table-tform" "-extract=Code" "-drop_titles"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}clang-{{[0-9]+}}"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}ptxas"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}fatbinary"
// CHK-ACTIONS: file-table-tform" "-replace=Code,Code"
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=nvptx64" "-kind=sycl"{{.*}}

// CHK-ACTIONS-WIN: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-pc-windows-msvc"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-emit-llvm-bc" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-target-feature" "+ptx42"{{.*}}
// CHK-ACTIONS-WIN: sycl-post-link{{.*}} "-split=auto"
// CHK-ACTIONS-WIN: file-table-tform" "-extract=Code" "-drop_titles"
// CHK-ACTIONS-WIN: llvm-foreach" {{.*}} "--" "{{.*}}clang-{{[0-9]+}}"
// CHK-ACTIONS-WIN: llvm-foreach" {{.*}} "--" "{{.*}}ptxas"
// CHK-ACTIONS-WIN: llvm-foreach" {{.*}} "--" "{{.*}}fatbinary"
// CHK-ACTIONS-WIN: file-table-tform" "-replace=Code,Code"
// CHK-ACTIONS-WIN-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS-WIN: clang-offload-wrapper"{{.*}} "-host=x86_64-pc-windows-msvc" "-target=nvptx64" "-kind=sycl"{{.*}}

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -std=c++11 \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/lib/nvidiacl \
// RUN: --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
//
// TODO: Enable for clang_cl once device lib linking works for clang_cl
//
// CHK-PHASES-NO-CC: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES-NO-CC: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 3: input, "{{.*}}", c++, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 5: compiler, {4}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 6: offload, "host-sycl (x86_64-{{.*}})" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {5}, c++-cpp-output
// CHK-PHASES-NO-CC: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES-NO-CC: 10: linker, {5}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 11: input, "{{.*}}libsycl-itt-user-wrappers.o", object
// CHK-PHASES-NO-CC: 12: clang-offload-unbundler, {11}, object
// CHK-PHASES-NO-CC: 13: offload, " (nvptx64-nvidia-cuda)" {12}, object
// CHK-PHASES-NO-CC: 14: input, "{{.*}}libsycl-itt-compiler-wrappers.o", object
// CHK-PHASES-NO-CC: 15: clang-offload-unbundler, {14}, object
// CHK-PHASES-NO-CC: 16: offload, " (nvptx64-nvidia-cuda)" {15}, object
// CHK-PHASES-NO-CC: 17: input, "{{.*}}libsycl-itt-stubs.o", object
// CHK-PHASES-NO-CC: 18: clang-offload-unbundler, {17}, object
// CHK-PHASES-NO-CC: 19: offload, " (nvptx64-nvidia-cuda)" {18}, object
// CHK-PHASES-NO-CC: 20: input, "{{.*}}nvidiacl{{.*}}", ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 21: input, "{{.*}}libdevice{{.*}}", ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 22: linker, {10, 13, 16, 19, 20, 21}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 23: sycl-post-link, {22}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 24: file-table-tform, {23}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 25: backend, {24}, assembler, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 26: assembler, {25}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 27: linker, {25, 26}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 28: foreach, {24, 27}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 29: file-table-tform, {23, 28}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 30: clang-offload-wrapper, {29}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 31: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {30}, object
// CHK-PHASES-NO-CC: 32: linker, {9, 31}, image, (host-sycl)
//
/// Check phases specifying a compute capability.
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -std=c++11 \
// RUN: -target x86_64-unknown-linux-gnu -fsycl -fno-sycl-device-lib=all \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/lib/nvidiacl \
// RUN: --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES %s
//
// TODO: Enable for clang_cl once device lib linking works for clang_cl
//
// CHK-PHASES: 0: input, "{{.*}}", c++, (host-sycl)
// CHK-PHASES: 1: append-footer, {0}, c++, (host-sycl)
// CHK-PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 3: input, "{{.*}}", c++, (device-sycl, sm_35)
// CHK-PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES: 5: compiler, {4}, ir, (device-sycl, sm_35)
// CHK-PHASES: 6: offload, "host-sycl (x86_64-{{.*}})" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {5}, c++-cpp-output
// CHK-PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHK-PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHK-PHASES: 9: assembler, {8}, object, (host-sycl)
// CHK-PHASES: 10: linker, {5}, ir, (device-sycl, sm_35)
// CHK-PHASES: 11: input, "{{.*}}libsycl-itt-user-wrappers.o", object
// CHK-PHASES: 12: clang-offload-unbundler, {11}, object
// CHK-PHASES: 13: offload, " (nvptx64-nvidia-cuda)" {12}, object
// CHK-PHASES: 14: input, "{{.*}}libsycl-itt-compiler-wrappers.o", object
// CHK-PHASES: 15: clang-offload-unbundler, {14}, object
// CHK-PHASES: 16: offload, " (nvptx64-nvidia-cuda)" {15}, object
// CHK-PHASES: 17: input, "{{.*}}libsycl-itt-stubs.o", object
// CHK-PHASES: 18: clang-offload-unbundler, {17}, object
// CHK-PHASES: 19: offload, " (nvptx64-nvidia-cuda)" {18}, object
// CHK-PHASES: 20: input, "{{.*}}nvidiacl{{.*}}", ir, (device-sycl, sm_35)
// CHK-PHASES: 21: input, "{{.*}}libdevice{{.*}}", ir, (device-sycl, sm_35)
// CHK-PHASES: 22: linker, {10, 13, 16, 19, 20, 21}, ir, (device-sycl, sm_35)
// CHK-PHASES: 23: sycl-post-link, {22}, ir, (device-sycl, sm_35)
// CHK-PHASES: 24: file-table-tform, {23}, ir, (device-sycl, sm_35)
// CHK-PHASES: 25: backend, {24}, assembler, (device-sycl, sm_35)
// CHK-PHASES: 26: assembler, {25}, object, (device-sycl, sm_35)
// CHK-PHASES: 27: linker, {25, 26}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 28: foreach, {24, 27}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 29: file-table-tform, {23, 28}, tempfiletable, (device-sycl, sm_35)
// CHK-PHASES: 30: clang-offload-wrapper, {29}, object, (device-sycl, sm_35)
// CHK-PHASES: 31: offload, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {30}, object
// CHK-PHASES: 32: linker, {9, 31}, image, (host-sycl)

/// Check calling preprocessor only
// RUN: %clangxx -E -fsycl -fsycl-targets=nvptx64-nvidia-cuda -ccc-print-phases %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PREPROC %s
// CHK-PREPROC: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, sm_[[CUDA_VERSION:[0-9.]+]])
// CHK-PREPROC: 2: offload, "device-sycl (nvptx64-nvidia-cuda:sm_[[CUDA_VERSION]])" {1}, c++-cpp-output
// CHK-PREPROC: 4: compiler, {1}, none, (device-sycl, sm_[[CUDA_VERSION]])
//
// RUN: not %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/no/CUDA/path/here \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-CUDA-PATH-ERROR %s
//
// RUN: not %clang_cl -### -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/no/CUDA/path/here \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-CUDA-PATH-ERROR %s
//
// CHK-CUDA-PATH-ERROR: provide path to different CUDA installation via '--cuda-path', or pass '-nocudalib' to build without linking with libdevice
//
//
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/no/CUDA/path/here \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -nocudalib %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-CUDA-NO-LIB %s
//
// RUN: %clang_cl -### -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/no/CUDA/path/here \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -nocudalib %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-CUDA-NO-LIB %s
//
// CHK-CUDA-NO-LIB-NOT: provide path to different CUDA installation via '--cuda-path', or pass '-nocudalib' to build without linking with libdevice
//
