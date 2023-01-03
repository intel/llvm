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
// RUN: -target x86_64-unknown-linux-gnu -fsycl \
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
// CHK-PHASES-NO-CC: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES-NO-CC: 11: linker, {5}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 12: input, "{{.*}}libsycl-crt.o", object
// CHK-PHASES-NO-CC: 13: clang-offload-unbundler, {12}, object
// CHK-PHASES-NO-CC: 14: offload, " (nvptx64-nvidia-cuda)" {13}, object
// CHK-PHASES-NO-CC: 15: input, "{{.*}}libsycl-complex.o", object
// CHK-PHASES-NO-CC: 16: clang-offload-unbundler, {15}, object
// CHK-PHASES-NO-CC: 17: offload, " (nvptx64-nvidia-cuda)" {16}, object
// CHK-PHASES-NO-CC: 18: input, "{{.*}}libsycl-complex-fp64.o", object
// CHK-PHASES-NO-CC: 19: clang-offload-unbundler, {18}, object
// CHK-PHASES-NO-CC: 20: offload, " (nvptx64-nvidia-cuda)" {19}, object
// CHK-PHASES-NO-CC: 21: input, "{{.*}}libsycl-cmath.o", object
// CHK-PHASES-NO-CC: 22: clang-offload-unbundler, {21}, object
// CHK-PHASES-NO-CC: 23: offload, " (nvptx64-nvidia-cuda)" {22}, object
// CHK-PHASES-NO-CC: 24: input, "{{.*}}libsycl-cmath-fp64.o", object
// CHK-PHASES-NO-CC: 25: clang-offload-unbundler, {24}, object
// CHK-PHASES-NO-CC: 26: offload, " (nvptx64-nvidia-cuda)" {25}, object
// CHK-PHASES-NO-CC: 27: input, "{{.*}}libsycl-imf.o", object
// CHK-PHASES-NO-CC: 28: clang-offload-unbundler, {27}, object
// CHK-PHASES-NO-CC: 29: offload, " (nvptx64-nvidia-cuda)" {28}, object
// CHK-PHASES-NO-CC: 30: input, "{{.*}}libsycl-imf-fp64.o", object
// CHK-PHASES-NO-CC: 31: clang-offload-unbundler, {30}, object
// CHK-PHASES-NO-CC: 32: offload, " (nvptx64-nvidia-cuda)" {31}, object
// CHK-PHASES-NO-CC: 33: input, "{{.*}}libsycl-imf-bf16.o", object
// CHK-PHASES-NO-CC: 34: clang-offload-unbundler, {33}, object
// CHK-PHASES-NO-CC: 35: offload, " (nvptx64-nvidia-cuda)" {34}, object
// CHK-PHASES-NO-CC: 36: input, "{{.*}}libsycl-fallback-cassert.o", object
// CHK-PHASES-NO-CC: 37: clang-offload-unbundler, {36}, object
// CHK-PHASES-NO-CC: 38: offload, " (nvptx64-nvidia-cuda)" {37}, object
// CHK-PHASES-NO-CC: 39: input, "{{.*}}libsycl-fallback-cstring.o", object
// CHK-PHASES-NO-CC: 40: clang-offload-unbundler, {39}, object
// CHK-PHASES-NO-CC: 41: offload, " (nvptx64-nvidia-cuda)" {40}, object
// CHK-PHASES-NO-CC: 42: input, "{{.*}}libsycl-fallback-complex.o", object
// CHK-PHASES-NO-CC: 43: clang-offload-unbundler, {42}, object
// CHK-PHASES-NO-CC: 44: offload, " (nvptx64-nvidia-cuda)" {43}, object
// CHK-PHASES-NO-CC: 45: input, "{{.*}}libsycl-fallback-complex-fp64.o", object
// CHK-PHASES-NO-CC: 46: clang-offload-unbundler, {45}, object
// CHK-PHASES-NO-CC: 47: offload, " (nvptx64-nvidia-cuda)" {46}, object
// CHK-PHASES-NO-CC: 48: input, "{{.*}}libsycl-fallback-cmath.o", object
// CHK-PHASES-NO-CC: 49: clang-offload-unbundler, {48}, object
// CHK-PHASES-NO-CC: 50: offload, " (nvptx64-nvidia-cuda)" {49}, object
// CHK-PHASES-NO-CC: 51: input, "{{.*}}libsycl-fallback-cmath-fp64.o", object
// CHK-PHASES-NO-CC: 52: clang-offload-unbundler, {51}, object
// CHK-PHASES-NO-CC: 53: offload, " (nvptx64-nvidia-cuda)" {52}, object
// CHK-PHASES-NO-CC: 54: input, "{{.*}}libsycl-fallback-imf.o", object
// CHK-PHASES-NO-CC: 55: clang-offload-unbundler, {54}, object
// CHK-PHASES-NO-CC: 56: offload, " (nvptx64-nvidia-cuda)" {55}, object
// CHK-PHASES-NO-CC: 57: input, "{{.*}}libsycl-fallback-imf-fp64.o", object
// CHK-PHASES-NO-CC: 58: clang-offload-unbundler, {57}, object
// CHK-PHASES-NO-CC: 59: offload, " (nvptx64-nvidia-cuda)" {58}, object
// CHK-PHASES-NO-CC: 60: input, "{{.*}}libsycl-fallback-imf-bf16.o", object
// CHK-PHASES-NO-CC: 61: clang-offload-unbundler, {60}, object
// CHK-PHASES-NO-CC: 62: offload, " (nvptx64-nvidia-cuda)" {61}, object
// CHK-PHASES-NO-CC: 63: input, "{{.*}}libsycl-itt-user-wrappers.o", object
// CHK-PHASES-NO-CC: 64: clang-offload-unbundler, {63}, object
// CHK-PHASES-NO-CC: 65: offload, " (nvptx64-nvidia-cuda)" {64}, object
// CHK-PHASES-NO-CC: 66: input, "{{.*}}libsycl-itt-compiler-wrappers.o", object
// CHK-PHASES-NO-CC: 67: clang-offload-unbundler, {66}, object
// CHK-PHASES-NO-CC: 68: offload, " (nvptx64-nvidia-cuda)" {67}, object
// CHK-PHASES-NO-CC: 69: input, "{{.*}}libsycl-itt-stubs.o", object
// CHK-PHASES-NO-CC: 70: clang-offload-unbundler, {69}, object
// CHK-PHASES-NO-CC: 71: offload, " (nvptx64-nvidia-cuda)" {70}, object
// CHK-PHASES-NO-CC: 72: input, "{{.*}}nvidiacl{{.*}}", ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 73: input, "{{.*}}libdevice{{.*}}", ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 74: linker, {11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 72, 73}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 75: sycl-post-link, {74}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 76: file-table-tform, {75}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 77: backend, {76}, assembler, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 78: assembler, {77}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 79: linker, {77, 78}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 80: foreach, {76, 79}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 81: file-table-tform, {75, 80}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 82: clang-offload-wrapper, {81}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 83: offload, "host-sycl (x86_64-{{.*}})" {10}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {82}, image
//
//
/// Check phases specifying a compute capability.
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -std=c++11 \
// RUN: -target x86_64-unknown-linux-gnu -fsycl \
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
// CHK-PHASES: 10: linker, {9}, image, (host-sycl)
// CHK-PHASES: 11: linker, {5}, ir, (device-sycl, sm_35)
// CHK-PHASES: 12: input, "{{.*}}libsycl-crt.o", object
// CHK-PHASES: 13: clang-offload-unbundler, {12}, object
// CHK-PHASES: 14: offload, " (nvptx64-nvidia-cuda)" {13}, object
// CHK-PHASES: 15: input, "{{.*}}libsycl-complex.o", object
// CHK-PHASES: 16: clang-offload-unbundler, {15}, object
// CHK-PHASES: 17: offload, " (nvptx64-nvidia-cuda)" {16}, object
// CHK-PHASES: 18: input, "{{.*}}libsycl-complex-fp64.o", object
// CHK-PHASES: 19: clang-offload-unbundler, {18}, object
// CHK-PHASES: 20: offload, " (nvptx64-nvidia-cuda)" {19}, object
// CHK-PHASES: 21: input, "{{.*}}libsycl-cmath.o", object
// CHK-PHASES: 22: clang-offload-unbundler, {21}, object
// CHK-PHASES: 23: offload, " (nvptx64-nvidia-cuda)" {22}, object
// CHK-PHASES: 24: input, "{{.*}}libsycl-cmath-fp64.o", object
// CHK-PHASES: 25: clang-offload-unbundler, {24}, object
// CHK-PHASES: 26: offload, " (nvptx64-nvidia-cuda)" {25}, object
// CHK-PHASES: 27: input, "{{.*}}libsycl-imf.o", object
// CHK-PHASES: 28: clang-offload-unbundler, {27}, object
// CHK-PHASES: 29: offload, " (nvptx64-nvidia-cuda)" {28}, object
// CHK-PHASES: 30: input, "{{.*}}libsycl-imf-fp64.o", object
// CHK-PHASES: 31: clang-offload-unbundler, {30}, object
// CHK-PHASES: 32: offload, " (nvptx64-nvidia-cuda)" {31}, object
// CHK-PHASES: 33: input, "{{.*}}libsycl-imf-bf16.o", object
// CHK-PHASES: 34: clang-offload-unbundler, {33}, object
// CHK-PHASES: 35: offload, " (nvptx64-nvidia-cuda)" {34}, object
// CHK-PHASES: 36: input, "{{.*}}libsycl-fallback-cassert.o", object
// CHK-PHASES: 37: clang-offload-unbundler, {36}, object
// CHK-PHASES: 38: offload, " (nvptx64-nvidia-cuda)" {37}, object
// CHK-PHASES: 39: input, "{{.*}}libsycl-fallback-cstring.o", object
// CHK-PHASES: 40: clang-offload-unbundler, {39}, object
// CHK-PHASES: 41: offload, " (nvptx64-nvidia-cuda)" {40}, object
// CHK-PHASES: 42: input, "{{.*}}libsycl-fallback-complex.o", object
// CHK-PHASES: 43: clang-offload-unbundler, {42}, object
// CHK-PHASES: 44: offload, " (nvptx64-nvidia-cuda)" {43}, object
// CHK-PHASES: 45: input, "{{.*}}libsycl-fallback-complex-fp64.o", object
// CHK-PHASES: 46: clang-offload-unbundler, {45}, object
// CHK-PHASES: 47: offload, " (nvptx64-nvidia-cuda)" {46}, object
// CHK-PHASES: 48: input, "{{.*}}libsycl-fallback-cmath.o", object
// CHK-PHASES: 49: clang-offload-unbundler, {48}, object
// CHK-PHASES: 50: offload, " (nvptx64-nvidia-cuda)" {49}, object
// CHK-PHASES: 51: input, "{{.*}}libsycl-fallback-cmath-fp64.o", object
// CHK-PHASES: 52: clang-offload-unbundler, {51}, object
// CHK-PHASES: 53: offload, " (nvptx64-nvidia-cuda)" {52}, object
// CHK-PHASES: 54: input, "{{.*}}libsycl-fallback-imf.o", object
// CHK-PHASES: 55: clang-offload-unbundler, {54}, object
// CHK-PHASES: 56: offload, " (nvptx64-nvidia-cuda)" {55}, object
// CHK-PHASES: 57: input, "{{.*}}libsycl-fallback-imf-fp64.o", object
// CHK-PHASES: 58: clang-offload-unbundler, {57}, object
// CHK-PHASES: 59: offload, " (nvptx64-nvidia-cuda)" {58}, object
// CHK-PHASES: 60: input, "{{.*}}libsycl-fallback-imf-bf16.o", object
// CHK-PHASES: 61: clang-offload-unbundler, {60}, object
// CHK-PHASES: 62: offload, " (nvptx64-nvidia-cuda)" {61}, object
// CHK-PHASES: 63: input, "{{.*}}libsycl-itt-user-wrappers.o", object
// CHK-PHASES: 64: clang-offload-unbundler, {63}, object
// CHK-PHASES: 65: offload, " (nvptx64-nvidia-cuda)" {64}, object
// CHK-PHASES: 66: input, "{{.*}}libsycl-itt-compiler-wrappers.o", object
// CHK-PHASES: 67: clang-offload-unbundler, {66}, object
// CHK-PHASES: 68: offload, " (nvptx64-nvidia-cuda)" {67}, object
// CHK-PHASES: 69: input, "{{.*}}libsycl-itt-stubs.o", object
// CHK-PHASES: 70: clang-offload-unbundler, {69}, object
// CHK-PHASES: 71: offload, " (nvptx64-nvidia-cuda)" {70}, object
// CHK-PHASES: 72: input, "{{.*}}nvidiacl{{.*}}", ir, (device-sycl, sm_35)
// CHK-PHASES: 73: input, "{{.*}}libdevice{{.*}}", ir, (device-sycl, sm_35)
// CHK-PHASES: 74: linker, {11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 72, 73}, ir, (device-sycl, sm_35)
// CHK-PHASES: 75: sycl-post-link, {74}, ir, (device-sycl, sm_35)
// CHK-PHASES: 76: file-table-tform, {75}, ir, (device-sycl, sm_35)
// CHK-PHASES: 77: backend, {76}, assembler, (device-sycl, sm_35)
// CHK-PHASES: 78: assembler, {77}, object, (device-sycl, sm_35)
// CHK-PHASES: 79: linker, {77, 78}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 80: foreach, {76, 79}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 81: file-table-tform, {75, 80}, tempfiletable, (device-sycl, sm_35)
// CHK-PHASES: 82: clang-offload-wrapper, {81}, object, (device-sycl, sm_35)
// CHK-PHASES: 83: offload, "host-sycl (x86_64-{{.*}})" {10}, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {82}, image

/// Check calling preprocessor only
// RUN: %clangxx -E -fsycl -fsycl-targets=nvptx64-nvidia-cuda -ccc-print-phases %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PREPROC %s
// CHK-PREPROC: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, sm_[[CUDA_VERSION:[0-9.]+]])
// CHK-PREPROC: 2: offload, "device-sycl (nvptx64-nvidia-cuda:sm_[[CUDA_VERSION]])" {1}, c++-cpp-output
// CHK-PREPROC: 4: compiler, {1}, none, (device-sycl, sm_[[CUDA_VERSION]])
//
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/no/CUDA/path/here \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-CUDA-PATH-ERROR %s
//
// RUN: %clang_cl -### -fsycl \
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
