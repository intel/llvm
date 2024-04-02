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
// NVIDIA_CHECK_PHASES: 1: append-footer, {0}, c++, (host-sycl)
// NVIDIA_CHECK_PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// NVIDIA_CHECK_PHASES: 3: input, "[[INPUT]]", c++, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 5: compiler, {4}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {5}, c++-cpp-output
// NVIDIA_CHECK_PHASES: 7: compiler, {6}, ir, (host-sycl)
// NVIDIA_CHECK_PHASES: 8: backend, {7}, assembler, (host-sycl)
// NVIDIA_CHECK_PHASES: 9: assembler, {8}, object, (host-sycl)
// NVIDIA_CHECK_PHASES: 10: linker, {5}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 11: sycl-post-link, {10}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 12: file-table-tform, {11}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 13: backend, {12}, assembler, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 14: assembler, {13}, object, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 15: linker, {13, 14}, cuda-fatbin, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 16: foreach, {12, 15}, cuda-fatbin, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 17: file-table-tform, {11, 16}, tempfiletable, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 18: clang-offload-wrapper, {17}, object, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 19: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {18}, object
// NVIDIA_CHECK_PHASES: 20: linker, {9, 19}, image, (host-sycl)
