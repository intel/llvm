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
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_11_2_0 -### %s 2>&1 | \
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
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=rpl_s \
// RUN:             -DMAC_STR=RPL_S
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
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g11 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g11 \
// RUN:             -DMAC_STR=ACM_G11
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_acm_g12 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=acm_g12 \
// RUN:             -DMAC_STR=ACM_G12
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE,MACRO -DDEV_STR=pvc -DMAC_STR=PVC
// MACRO: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// MACRO: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// DEVICE: ocloc{{.*}} "-device" "[[DEV_STR]]"

/// -fsycl-targets=spir64_x86_64 should set a specific macro
// RUN: %clangxx -c -fsycl -fsycl-targets=spir64_x86_64 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MACRO_X86_64
// RUN: %clang_cl -c -fsycl -fsycl-targets=spir64_x86_64 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=MACRO_X86_64
// MACRO_X86_64: clang{{.*}} "-triple" "spir64_x86_64-unknown-unknown"
// MACRO_X86_64: "-D__SYCL_TARGET_INTEL_X86_64__"

/// test for invalid arch
// RUN: %clangxx -c -fsycl -fsycl-targets=intel_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_INPUT
// RUN: %clang_cl -c -fsycl -fsycl-targets=intel_gpu_bad -### %s 2>&1 | \
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

