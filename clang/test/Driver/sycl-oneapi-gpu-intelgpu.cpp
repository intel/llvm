/// Tests the behaviors of using -fsycl-targets=intel_gpu*

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_bdw -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=bdw -DMAC_STR=BDW
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_8_0_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=bdw -DMAC_STR=BDW
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_skl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=skl -DMAC_STR=SKL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_0_9 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=skl -DMAC_STR=SKL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_kbl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=kbl -DMAC_STR=KBL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_1_9 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=kbl -DMAC_STR=KBL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_cfl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=cfl -DMAC_STR=CFL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_2_9 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=cfl -DMAC_STR=CFL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_apl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=apl -DMAC_STR=APL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_3_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=apl -DMAC_STR=APL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_bxt -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=apl -DMAC_STR=APL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_glk -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=glk -DMAC_STR=GLK
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_4_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=glk -DMAC_STR=GLK
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_whl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=whl -DMAC_STR=WHL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_5_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=whl -DMAC_STR=WHL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_aml -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=aml -DMAC_STR=AML
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_6_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=aml -DMAC_STR=AML
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_cml -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=cml -DMAC_STR=CML
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_9_7_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=cml -DMAC_STR=CML
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_icllp -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=icllp \
// RUN:             -DMAC_STR=ICLLP
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_11_0_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=icllp \
// RUN:             -DMAC_STR=ICLLP
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_ehl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=ehl -DMAC_STR=EHL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_jsl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=ehl -DMAC_STR=EHL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_tgllp -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=tgllp \
// RUN:             -DMAC_STR=TGLLP
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_12_0_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=tgllp \
// RUN:             -DMAC_STR=TGLLP
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_rkl -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=rkl -DMAC_STR=RKL
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_adl_s -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=adl_s \
// RUN:             -DMAC_STR=ADL_S
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_rpl_s -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=adl_s \
// RUN:             -DMAC_STR=ADL_S
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_adl_p -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=adl_p \
// RUN:             -DMAC_STR=ADL_P
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_adl_n -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=adl_n \
// RUN:             -DMAC_STR=ADL_N
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=dg1 -DMAC_STR=DG1
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_12_10_0 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=dg1 -DMAC_STR=DG1
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g10 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g10 \
// RUN:             -DMAC_STR=ACM_G10
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g10 \
// RUN:             -DMAC_STR=ACM_G10
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g11 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g11 \
// RUN:             -DMAC_STR=ACM_G11
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g11 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g11 \
// RUN:             -DMAC_STR=ACM_G11
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g12 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g12 \
// RUN:             -DMAC_STR=ACM_G12
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g12 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g12 \
// RUN:             -DMAC_STR=ACM_G12
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=pvc -DMAC_STR=PVC
// MACRO: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// MACRO: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// MACRO: clang{{.*}} "-fsycl-is-host"
// MACRO: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// DEVICE: ocloc{{.*}} "-device" "[[DEV_STR]]"

/// -fsycl-targets=spir64_x86_64 should set a specific macro
// RUN: %clangxx -c -fsycl -fsycl-targets=spir64_x86_64 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MACRO_X86_64
// RUN: %clang_cl -c -fsycl -fsycl-targets=spir64_x86_64 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MACRO_X86_64
// MACRO_X86_64: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown"
// MACRO_X86_64: "-D__SYCL_TARGET_INTEL_X86_64__"
// MACRO_X86_64: clang{{.*}} "-fsycl-is-host"
// MACRO_X86_64: "-D__SYCL_TARGET_INTEL_X86_64__"

/// test for invalid intel arch
// RUN: not %clangxx -c -fsycl -fsycl-targets=intel_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_INPUT
// RUN: not %clang_cl -c -fsycl -fsycl-targets=intel_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_INPUT
// BAD_INPUT: error: SYCL target is invalid: 'intel_gpu_bad'

/// Test for proper creation of fat object
// RUN: %clangxx -c -fsycl -fsycl-targets=intel_gpu_skl \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=FATO
// FATO: clang-offload-bundler{{.*}} "-type=o"
// FATO: "-targets=sycl-spir64_gen-unknown-unknown-skl,host-x86_64-unknown-linux-gnu"

/// Test for proper consumption of fat object
// RUN: touch %t.o
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_skl \
// RUN:   -target x86_64-unknown-linux-gnu -### %t.o 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CONSUME_FAT
// CONSUME_FAT: clang-offload-bundler{{.*}} "-type=o"
// CONSUME_FAT: "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-skl"
// CONSUME_FAT: "-unbundle" "-allow-missing-bundles"

/// Test phases, BoundArch settings used for -device target. Additional
/// offload action used for compilation and backend compilation.
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_skl -fno-sycl-device-lib=all \
// RUN:   -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK_PHASES
// CHECK_PHASES: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHECK_PHASES: 1: append-footer, {0}, c++, (host-sycl)
// CHECK_PHASES: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHECK_PHASES: 3: input, "[[INPUT]]", c++, (device-sycl, skl)
// CHECK_PHASES: 4: preprocessor, {3}, c++-cpp-output, (device-sycl, skl)
// CHECK_PHASES: 5: compiler, {4}, ir, (device-sycl, skl)
// CHECK_PHASES: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_gen-unknown-unknown:skl)" {5}, c++-cpp-output
// CHECK_PHASES: 7: compiler, {6}, ir, (host-sycl)
// CHECK_PHASES: 8: backend, {7}, assembler, (host-sycl)
// CHECK_PHASES: 9: assembler, {8}, object, (host-sycl)
// CHECK_PHASES: 10: linker, {5}, ir, (device-sycl, skl)
// CHECK_PHASES: 11: sycl-post-link, {10}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES: 12: file-table-tform, {11}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES: 13: llvm-spirv, {12}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES: 14: backend-compiler, {13}, image, (device-sycl, skl)
// CHECK_PHASES: 15: file-table-tform, {11, 14}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES: 16: clang-offload-wrapper, {15}, object, (device-sycl, skl)
// CHECK_PHASES: 17: offload, "device-sycl (spir64_gen-unknown-unknown:skl)" {16}, object
// CHECK_PHASES: 18: linker, {9, 17}, image, (host-sycl)

/// Check that ocloc and macro settings only occur for the expected toolchains
/// when mixing spir64_gen and intel_gpu
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device skl" \
// RUN:   -fno-sycl-device-lib=all -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK_TOOLS_MIX
// CHECK_TOOLS_MIX: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// CHECK_TOOLS_MIX-NOT: "-D__SYCL_TARGET_INTEL_GPU{{.*}}"
// CHECK_TOOLS_MIX: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// CHECK_TOOLS_MIX: "-D__SYCL_TARGET_INTEL_GPU_DG1__"
// CHECK_TOOLS_MIX: ocloc{{.*}} "-device" "dg1"
// CHECK_TOOLS_MIX: ocloc{{.*}} "-device" "skl"

/// Test phases when using both spir64_gen and intel_gpu*
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_skl,spir64_gen \
// RUN:   -fno-sycl-device-lib=all -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -ccc-print-phases %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK_PHASES_MIX
// CHECK_PHASES_MIX: 0: input, "[[INPUT:.+\.cpp]]", c++, (host-sycl)
// CHECK_PHASES_MIX: 1: append-footer, {0}, c++, (host-sycl)
// CHECK_PHASES_MIX: 2: preprocessor, {1}, c++-cpp-output, (host-sycl)
// CHECK_PHASES_MIX: 3: input, "[[INPUT]]", c++, (device-sycl)
// CHECK_PHASES_MIX: 4: preprocessor, {3}, c++-cpp-output, (device-sycl)
// CHECK_PHASES_MIX: 5: compiler, {4}, ir, (device-sycl)
// CHECK_PHASES_MIX: 6: offload, "host-sycl (x86_64-unknown-linux-gnu)" {2}, "device-sycl (spir64_gen-unknown-unknown)" {5}, c++-cpp-output
// CHECK_PHASES_MIX: 7: compiler, {6}, ir, (host-sycl)
// CHECK_PHASES_MIX: 8: backend, {7}, assembler, (host-sycl)
// CHECK_PHASES_MIX: 9: assembler, {8}, object, (host-sycl)
// CHECK_PHASES_MIX: 10: input, "[[INPUT]]", c++, (device-sycl, skl)
// CHECK_PHASES_MIX: 11: preprocessor, {10}, c++-cpp-output, (device-sycl, skl)
// CHECK_PHASES_MIX: 12: compiler, {11}, ir, (device-sycl, skl)
// CHECK_PHASES_MIX: 13: linker, {12}, ir, (device-sycl, skl)
// CHECK_PHASES_MIX: 14: sycl-post-link, {13}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES_MIX: 15: file-table-tform, {14}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES_MIX: 16: llvm-spirv, {15}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES_MIX: 17: backend-compiler, {16}, image, (device-sycl, skl)
// CHECK_PHASES_MIX: 18: file-table-tform, {14, 17}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES_MIX: 19: clang-offload-wrapper, {18}, object, (device-sycl, skl)
// CHECK_PHASES_MIX: 20: offload, "device-sycl (spir64_gen-unknown-unknown:skl)" {19}, object
// CHECK_PHASES_MIX: 21: linker, {5}, ir, (device-sycl)
// CHECK_PHASES_MIX: 22: sycl-post-link, {21}, tempfiletable, (device-sycl)
// CHECK_PHASES_MIX: 23: file-table-tform, {22}, tempfilelist, (device-sycl)
// CHECK_PHASES_MIX: 24: llvm-spirv, {23}, tempfilelist, (device-sycl)
// CHECK_PHASES_MIX: 25: backend-compiler, {24}, image, (device-sycl)
// CHECK_PHASES_MIX: 26: file-table-tform, {22, 25}, tempfiletable, (device-sycl)
// CHECK_PHASES_MIX: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHECK_PHASES_MIX: 28: offload, "device-sycl (spir64_gen-unknown-unknown)" {27}, object
// CHECK_PHASES_MIX: 29: linker, {9, 20, 28}, image, (host-sycl)

/// Check that ocloc backend option settings only occur for the expected
/// toolchains when mixing spir64_gen and intel_gpu
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spir64_gen,intel_gpu_skl \
// RUN:   -Xsycl-target-backend=spir64_gen "-device skl -DSKL" \
// RUN:   -Xsycl-target-backend=intel_gpu_dg1 "-DDG1" \
// RUN:   -Xsycl-target-backend=intel_gpu_skl "-DSKL2" \
// RUN:   -fno-sycl-device-lib=all -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK_TOOLS_BEOPTS
// CHECK_TOOLS_BEOPTS: ocloc{{.*}} "-device" "dg1"{{.*}}"-DDG1"
// CHECK_TOOLS_BEOPTS: ocloc{{.*}} "-device" "skl"{{.*}}"-DSKL"
// CHECK_TOOLS_BEOPTS: ocloc{{.*}} "-device" "skl"{{.*}}"-DSKL2"

/// Check that ocloc backend option settings only occur for the expected
/// toolchains when mixing intel_gpu and non-spir64_gen targets
// RUN: %clangxx -fsycl-targets=intel_gpu_dg1,spir64_x86_64,intel_gpu_skl \
// RUN:   -fsycl -Xsycl-target-backend=spir64_x86_64 "-DCPU" \
// RUN:   -Xsycl-target-backend=intel_gpu_dg1 "-DDG1" \
// RUN:   -Xsycl-target-backend=intel_gpu_skl "-DSKL2" \
// RUN:   -fno-sycl-device-lib=all -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK_TOOLS_BEOPTS_MIX
// CHECK_TOOLS_BEOPTS_MIX: ocloc{{.*}} "-device" "dg1"{{.*}}"-DDG1"
// CHECK_TOOLS_BEOPTS_MIX: opencl-aot{{.*}} "-DCPU"
// CHECK_TOOLS_BEOPTS_MIX-NOT: "-DDG1"
// CHECK_TOOLS_BEOPTS_MIX: ocloc{{.*}} "-device" "skl"{{.*}}"-DSKL2"
