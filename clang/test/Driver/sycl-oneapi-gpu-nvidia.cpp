/// Tests the behaviors of using -fsycl-targets=nvidia_gpu*

// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_50 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_50 -DMAC_STR=SM_50
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_52 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_52 -DMAC_STR=SM_52
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_53 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_53 -DMAC_STR=SM_53
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_60 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_60 -DMAC_STR=SM_60
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_61 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_61 -DMAC_STR=SM_61
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_62 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_62 -DMAC_STR=SM_62
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_70 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_70 -DMAC_STR=SM_70
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_72 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_72 -DMAC_STR=SM_72
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_75 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_75 -DMAC_STR=SM_75
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_80 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_80 -DMAC_STR=SM_80
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_86 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_86 -DMAC_STR=SM_86
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_87 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_87 -DMAC_STR=SM_87
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_89 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_89 -DMAC_STR=SM_89
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_90 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_90 -DMAC_STR=SM_90
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_90a -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_90a -DMAC_STR=SM_90A
// MACRO_NVIDIA: clang{{.*}}  "-fsycl-is-host"
// MACRO_NVIDIA: "-D__SYCL_TARGET_NVIDIA_GPU_[[MAC_STR]]__"
// MACRO_NVIDIA: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"
// DEVICE_NVIDIA: llvm-foreach{{.*}} "--gpu-name" "[[DEV_STR]]"


/// test for invalid nvidia arch
// RUN: not %clangxx -c -fsycl -fsycl-targets=nvidia_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_NVIDIA_INPUT
// RUN: not %clang_cl -c -fsycl -fsycl-targets=nvidia_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_NVIDIA_INPUT
// BAD_NVIDIA_INPUT: error: SYCL target is invalid: 'nvidia_gpu_bad'

// Check if the partial SYCL triple for NVidia GPUs translate to the full string.
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvptx64 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NVIDIA-TRIPLE
// NVIDIA-TRIPLE: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"

// Check if SYCL triples with 'Environment' component are rejected for NVidia GPUs.
// RUN: not %clangxx -c -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_TARGET_TRIPLE_ENV
// RUN: not %clang_cl -c -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_TARGET_TRIPLE_ENV
// BAD_TARGET_TRIPLE_ENV: error: SYCL target is invalid: 'nvptx64-nvidia-cuda-sycl'

// Check for invalid SYCL triple for NVidia GPUs.
// RUN: not %clangxx -c -fsycl -fsycl-targets=nvptx-nvidia-cuda -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_TARGET_TRIPLE
// RUN: not %clang_cl -c -fsycl -fsycl-targets=nvptx-nvidia-cuda -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_TARGET_TRIPLE
// BAD_TARGET_TRIPLE: error: SYCL target is invalid: 'nvptx-nvidia-cuda'

/// Test for proper creation of fat object
// RUN: %clangxx -c -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_50 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NVIDIA_FATO
// NVIDIA_FATO: clang-offload-bundler{{.*}} "-type=o"
// NVIDIA_FATO: "-targets=sycl-nvptx64-nvidia-cuda-sm_50,host-x86_64-unknown-linux-gnu"

/// Test for proper consumption of fat object
// RUN: touch %t.o
// RUN: %clangxx -fsycl -nocudalib -fsycl-targets=nvidia_gpu_sm_50 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc \
// RUN:   -target x86_64-unknown-linux-gnu -### %t.o 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NVIDIA_CONSUME_FAT
// NVIDIA_CONSUME_FAT: clang-offload-bundler{{.*}} "-type=o"
// NVIDIA_CONSUME_FAT: "-targets=host-x86_64-unknown-linux-gnu,sycl-nvptx64-nvidia-cuda-sm_50"
// NVIDIA_CONSUME_FAT: "-unbundle" "-allow-missing-bundles"

/// NVIDIA Test phases, BoundArch settings used for -device target. Additional
/// offload action used for compilation and backend compilation.
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_50 -fno-sycl-device-lib=all \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NVIDIA_CHECK_PHASES
// NVIDIA_CHECK_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// NVIDIA_CHECK_PHASES: 1: preprocessor, {0}, c++-cpp-output, (host-sycl)
// NVIDIA_CHECK_PHASES: 2: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 3: preprocessor, {2}, c++-cpp-output, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 4: compiler, {3}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 5: offload, "host-sycl (x86_64-unknown-linux-gnu)" {1}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {4}, c++-cpp-output
// NVIDIA_CHECK_PHASES: 6: compiler, {5}, ir, (host-sycl)
// NVIDIA_CHECK_PHASES: 7: backend, {6}, assembler, (host-sycl)
// NVIDIA_CHECK_PHASES: 8: assembler, {7}, object, (host-sycl)
// NVIDIA_CHECK_PHASES: 9: linker, {4}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 10: sycl-post-link, {9}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 11: file-table-tform, {10}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 12: backend, {11}, assembler, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 13: assembler, {12}, object, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 14: linker, {12, 13}, cuda-fatbin, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 15: foreach, {11, 14}, cuda-fatbin, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 16: file-table-tform, {10, 15}, tempfiletable, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 17: clang-offload-wrapper, {16}, object, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 18: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {17}, object
// NVIDIA_CHECK_PHASES: 19: linker, {8, 18}, image, (host-sycl)
