/// Tests the behaviors of using -fsycl-targets=amd_gpu*

// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx700 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx700 -DMAC_STR=GFX700
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx701 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx701 -DMAC_STR=GFX701
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx702 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx702 -DMAC_STR=GFX702
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx801 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx801 -DMAC_STR=GFX801
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx802 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx802 -DMAC_STR=GFX802
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx803 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx803 -DMAC_STR=GFX803
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx805 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx805 -DMAC_STR=GFX805
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx810 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx810 -DMAC_STR=GFX810
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx900 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx900 -DMAC_STR=GFX900
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx902 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx902 -DMAC_STR=GFX902
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx904 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx904 -DMAC_STR=GFX904
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx906 -DMAC_STR=GFX906
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx908 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx908 -DMAC_STR=GFX908
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx909 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx909 -DMAC_STR=GFX909
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx90a \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx90a -DMAC_STR=GFX90A
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx90c \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx90c -DMAC_STR=GFX90C
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx940 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx940 -DMAC_STR=GFX940
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx941 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx941 -DMAC_STR=GFX941
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx942 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx942 -DMAC_STR=GFX942
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1010 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1010 -DMAC_STR=GFX1010
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1011 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1011 -DMAC_STR=GFX1011
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1012 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1012 -DMAC_STR=GFX1012
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1013 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1013 -DMAC_STR=GFX1013
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1030 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1030 -DMAC_STR=GFX1030
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1031 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1031 -DMAC_STR=GFX1031
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1032 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1032 -DMAC_STR=GFX1032
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1033 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1033 -DMAC_STR=GFX1033
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1034 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1034 -DMAC_STR=GFX1034
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1035 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1035 -DMAC_STR=GFX1035
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1036 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1036 -DMAC_STR=GFX1036
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1100 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1100 -DMAC_STR=GFX1100
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1101 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1101 -DMAC_STR=GFX1101
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1102 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1102 -DMAC_STR=GFX1102
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1103 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1103 -DMAC_STR=GFX1103
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1150 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1150 -DMAC_STR=GFX1150
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1151 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1151 -DMAC_STR=GFX1151
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1200 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1200 -DMAC_STR=GFX1200
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx1201 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1201 -DMAC_STR=GFX1201
// MACRO_AMD: clang{{.*}} "-triple" "amdgcn-amd-amdhsa"
// MACRO_AMD: "-D__SYCL_TARGET_AMD_GPU_[[MAC_STR]]__"
// MACRO_AMD: clang{{.*}} "-fsycl-is-host"
// MACRO_AMD: "-D__SYCL_TARGET_AMD_GPU_[[MAC_STR]]__"
// DEVICE_AMD: clang-offload-wrapper{{.*}} "-compile-opts=--offload-arch=[[DEV_STR]]{{.*}}"

/// test for invalid amd arch
// RUN: not %clangxx -c -fsycl -fsycl-targets=amd_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_AMD_INPUT
// RUN: not %clang_cl -c -fsycl -fsycl-targets=amd_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_AMD_INPUT
// BAD_AMD_INPUT: error: SYCL target is invalid: 'amd_gpu_bad'

/// Test for proper creation of fat object
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx700 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=AMD_FATO
// AMD_FATO: clang-offload-bundler{{.*}} "-type=o"
// AMD_FATO: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx700"

/// Test for proper consumption of fat object
// RUN: touch %t.o
// RUN: %clangxx -fsycl -nogpulib -fsycl-targets=amd_gpu_gfx700 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc \
// RUN:   -target x86_64-unknown-linux-gnu -### %t.o 2>&1 | \
// RUN:   FileCheck %s --check-prefix=AMD_CONSUME_FAT
// AMD_CONSUME_FAT: clang-offload-bundler{{.*}} "-type=o"
// AMD_CONSUME_FAT: "-targets=host-x86_64-unknown-linux-gnu,sycl-amdgcn-amd-amdhsa-gfx700"
// AMD_CONSUME_FAT: "-unbundle" "-allow-missing-bundles"

/// AMD Test phases, BoundArch settings used for -device target. Additional
/// offload action used for compilation and backend compilation.
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx700 -fno-sycl-device-lib=all \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=AMD_CHECK_PHASES
// AMD_CHECK_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// AMD_CHECK_PHASES: 1: append-footer, {0}, c++, (host-sycl)
// AMD_CHECK_PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// AMD_CHECK_PHASES: 3: input, "[[INPUT]]", c++, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 5: compiler, {4}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (amdgcn-amd-amdhsa:gfx700)" {5}, c++-cpp-output
// AMD_CHECK_PHASES: 7: compiler, {6}, ir, (host-sycl)
// AMD_CHECK_PHASES: 8: backend, {7}, assembler, (host-sycl)
// AMD_CHECK_PHASES: 9: assembler, {8}, object, (host-sycl)
// AMD_CHECK_PHASES: 10: linker, {5}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 11: sycl-post-link, {10}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 12: file-table-tform, {11}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 13: backend, {12}, assembler, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 14: assembler, {13}, object, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 15: linker, {14}, image, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 16: linker, {15}, hip-fatbin, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 17: foreach, {12, 16}, hip-fatbin, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 18: file-table-tform, {11, 17}, tempfiletable, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 19: clang-offload-wrapper, {18}, object, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 20: offload, "device-sycl (amdgcn-amd-amdhsa:gfx700)" {19}, object
// AMD_CHECK_PHASES: 21: linker, {9, 20}, image, (host-sycl)

