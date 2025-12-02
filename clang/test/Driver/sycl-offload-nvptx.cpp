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

// CHK-ACTIONS: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-emit-llvm-bc" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-target-feature" "+ptx42"{{.*}} "-std=c++11"{{.*}}
// CHK-ACTIONS: sycl-post-link{{.*}} "-split=auto"
// CHK-ACTIONS: file-table-tform" "-extract=Code" "-drop_titles"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}clang-{{[0-9]+}}"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}ptxas"
// CHK-ACTIONS: llvm-foreach" {{.*}} "--" "{{.*}}fatbinary"
// CHK-ACTIONS: file-table-tform" "-replace=Code,Code"
// CHK-ACTIONS-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS: clang-offload-wrapper"{{.*}} "-host=x86_64-unknown-linux-gnu" "-target=nvptx64" "-kind=sycl"{{.*}}

// CHK-ACTIONS-WIN: "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "x86_64-pc-windows-msvc"{{.*}} "-fsycl-is-device"{{.*}} "-Wno-sycl-strict"{{.*}} "-sycl-std=2020" {{.*}} "-emit-llvm-bc" {{.*}} "-internal-isystem" "{{.*}}bin{{[/\\]+}}..{{[/\\]+}}include{{[/\\]+}}sycl{{[/\\]+}}stl_wrappers"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libspirv.bc"{{.*}} "-mlink-builtin-bitcode" "{{.*}}libdevice{{.*}}.10.bc"{{.*}} "-target-sdk-version=[[CUDA_VERSION:[0-9.]+]]"{{.*}} "-target-cpu" "sm_50"{{.*}} "-target-feature" "+ptx42"{{.*}}
// CHK-ACTIONS-WIN: sycl-post-link{{.*}} "-split=auto"
// CHK-ACTIONS-WIN: file-table-tform" "-extract=Code" "-drop_titles"
// CHK-ACTIONS-WIN: llvm-foreach" {{.*}} "--" "{{.*}}clang-{{[0-9]+}}"
// CHK-ACTIONS-WIN: llvm-foreach" {{.*}} "--" "{{.*}}ptxas"
// CHK-ACTIONS-WIN: llvm-foreach" {{.*}} "--" "{{.*}}fatbinary"
// CHK-ACTIONS-WIN: file-table-tform" "-replace=Code,Code"
// CHK-ACTIONS-WIN-NOT: "-mllvm -sycl-opt"
// CHK-ACTIONS-WIN: clang-offload-wrapper"{{.*}} "-host=[[HOST_TARGET:x86_64-pc-windows-msvc.*]]" "-target=nvptx64" "-kind=sycl"{{.*}}
// CHK-ACTIONS-WIN: clang{{.*}} "--target=[[HOST_TARGET]]" "-c"

/// Check phases w/out specifying a compute capability.
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -std=c++11 \
// RUN: -target x86_64-unknown-linux-gnu -fsycl --no-offloadlib \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda %s 2>&1 \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/lib/clang/resource_dir/lib/libclc/remangled-l32-signed_char.libspirv-nvptx64-nvidia-cuda.bc \
// RUN: --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN: | FileCheck -check-prefix=CHK-PHASES-NO-CC %s
//
// TODO: Enable for clang_cl once device lib linking works for clang_cl
//
// CHK-PHASES-NO-CC: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-PHASES-NO-CC: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES-NO-CC: 2: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 4: compiler, {3}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {4}, c++-cpp-output
// CHK-PHASES-NO-CC: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES-NO-CC: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES-NO-CC: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES-NO-CC: 9: linker, {4}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 10: input, "{{.*}}libspirv-nvptx64{{.*}}", ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 11: linker, {9, 10}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 12: sycl-post-link, {11}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 13: file-table-tform, {12}, ir, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 14: backend, {13}, assembler, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 15: assembler, {14}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 16: linker, {14, 15}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 17: foreach, {13, 16}, cuda-fatbin, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 18: file-table-tform, {12, 17}, tempfiletable, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 19: clang-offload-wrapper, {18}, object, (device-sycl, sm_50)
// CHK-PHASES-NO-CC: 20: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {19}, object
// CHK-PHASES-NO-CC: 21: linker, {8, 20}, image, (host-sycl)
//
/// Check phases specifying a compute capability.
// RUN: %clangxx -ccc-print-phases --sysroot=%S/Inputs/SYCL -std=c++11 \
// RUN: -target x86_64-unknown-linux-gnu -fsycl --no-offloadlib \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/lib/clang/resource_dir/lib/libclc/remangled-l32-signed_char.libspirv-nvptx64-nvidia-cuda.bc \
// RUN: --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN: -Xsycl-target-backend "--cuda-gpu-arch=sm_35" %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-PHASES %s
//
// TODO: Enable for clang_cl once device lib linking works for clang_cl
//
// CHK-PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHK-PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// CHK-PHASES: 2: input, "[[INPUT]]", c++, (device-sycl, sm_35)
// CHK-PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_35)
// CHK-PHASES: 4: compiler, {3}, ir, (device-sycl, sm_35)
// CHK-PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {4}, c++-cpp-output
// CHK-PHASES: 6: compiler, {5}, ir, (host-sycl)
// CHK-PHASES: 7: backend, {6}, assembler, (host-sycl)
// CHK-PHASES: 8: assembler, {7}, object, (host-sycl)
// CHK-PHASES: 9: linker, {4}, ir, (device-sycl, sm_35)
// CHK-PHASES: 10: input, "{{.*}}libspirv-nvptx64{{.*}}", ir, (device-sycl, sm_35)
// CHK-PHASES: 11: linker, {9, 10}, ir, (device-sycl, sm_35)
// CHK-PHASES: 12: sycl-post-link, {11}, ir, (device-sycl, sm_35)
// CHK-PHASES: 13: file-table-tform, {12}, ir, (device-sycl, sm_35)
// CHK-PHASES: 14: backend, {13}, assembler, (device-sycl, sm_35)
// CHK-PHASES: 15: assembler, {14}, object, (device-sycl, sm_35)
// CHK-PHASES: 16: linker, {14, 15}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 17: foreach, {13, 16}, cuda-fatbin, (device-sycl, sm_35)
// CHK-PHASES: 18: file-table-tform, {12, 17}, tempfiletable, (device-sycl, sm_35)
// CHK-PHASES: 19: clang-offload-wrapper, {18}, object, (device-sycl, sm_35)
// CHK-PHASES: 20: offload, "device-sycl (nvptx64-nvidia-cuda:sm_35)" {19}, object
// CHK-PHASES: 21: linker, {8, 20}, image, (host-sycl)

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

// Check that flags linked to fsycl-cuda-compatibility are set correctly
// RUN: %clangxx -### -std=c++11 -target x86_64-unknown-linux-gnu -fsycl -fsycl-cuda-compatibility \
// RUN: -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN: -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHK-ACTIONS-CUDA-COMPAT %s
// CHK-ACTIONS-CUDA-COMPAT: "-cc1" "-triple" "nvptx64-nvidia-cuda"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-cuda-compatibility"{{.*}} "-fdeclspec"{{.*}} "-fcuda-allow-variadic-functions"{{.*}} "-target-sdk-version=7.0"{{.*}} "-include" "__clang_cuda_runtime_wrapper.h"
// CHK-ACTIONS-CUDA-COMPAT: "-cc1" "-triple" "x86_64-unknown-linux-gnu"{{.*}} "-fsycl-is-host"{{.*}} "-fsycl-cuda-compatibility"{{.*}} "-fdeclspec"{{.*}} "-fcuda-allow-variadic-functions"{{.*}} "-aux-triple" "nvptx64-nvidia-cuda"{{.*}} "-target-sdk-version=7.0"{{.*}} "-include" "__clang_cuda_runtime_wrapper.h"
