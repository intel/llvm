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
// MACRO: clang{{.*}} "-fsycl-is-host"
// MACRO: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"

/// Tests the behaviors of using -fsycl-targets=nvidia_gpu*

// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_50 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_50 -DMAC_STR=SM_50
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_52 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_52 -DMAC_STR=SM_52
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_53 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_53 -DMAC_STR=SM_53
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_60 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_60 -DMAC_STR=SM_60
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_61 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_61 -DMAC_STR=SM_61
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_62 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_62 -DMAC_STR=SM_62
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_70 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_70 -DMAC_STR=SM_70
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_72 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_72 -DMAC_STR=SM_72
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_75 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_75 -DMAC_STR=SM_75
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_80 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_80 -DMAC_STR=SM_80
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_86 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_86 -DMAC_STR=SM_86
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_87 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_87 -DMAC_STR=SM_87
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_89 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_89 -DMAC_STR=SM_89
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_90 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_NVIDIA,MACRO_NVIDIA -DDEV_STR=sm_90 -DMAC_STR=SM_90
// MACRO_NVIDIA: clang{{.*}} "-triple" "nvptx64-nvidia-cuda"
// DEVICE_NVIDIA: llvm-foreach{{.*}} "--gpu-name" "[[DEV_STR]]"
// MACRO_NVIDIA: clang{{.*}}  "-fsycl-is-host"
// MACRO_NVIDIA: "-D__SYCL_TARGET_NVIDIA_GPU_[[MAC_STR]]__"

/// Tests the behaviors of using -fsycl-targets=amd_gpu*

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx700 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx700 -DMAC_STR=GFX700
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx701 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx701 -DMAC_STR=GFX701
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx702 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx702 -DMAC_STR=GFX702
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx801 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx801 -DMAC_STR=GFX801
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx802 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx802 -DMAC_STR=GFX802
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx803 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx803 -DMAC_STR=GFX803
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx805 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx805 -DMAC_STR=GFX805
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx810 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx810 -DMAC_STR=GFX810
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx900 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx900 -DMAC_STR=GFX900
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx902 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx902 -DMAC_STR=GFX902
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx904 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx904 -DMAC_STR=GFX904
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx906 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx906 -DMAC_STR=GFX906
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx908 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx908 -DMAC_STR=GFX908
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx90a -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx90a -DMAC_STR=GFX90A
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1010 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1010 -DMAC_STR=GFX1010
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1011 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1011 -DMAC_STR=GFX1011
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1012 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1012 -DMAC_STR=GFX1012
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1013 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1013 -DMAC_STR=GFX1013
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1030 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1030 -DMAC_STR=GFX1030
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1031 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1031 -DMAC_STR=GFX1031
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1032 -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEVICE_AMD,MACRO_AMD -DDEV_STR=gfx1032 -DMAC_STR=GFX1032
// MACRO_AMD: clang{{.*}} "-triple" "amdgcn-amd-amdhsa"
// MACRO_AMD: "-D__SYCL_TARGET_AMD_GPU_[[MAC_STR]]__"
// DEVICE_AMD: clang-offload-wrapper{{.*}} "-compile-opts=--offload-arch=[[DEV_STR]]{{.*}}"
// MACRO_AMD: clang{{.*}} "-fsycl-is-host"
// MACRO_AMD: "-D__SYCL_TARGET_AMD_GPU_[[MAC_STR]]__"

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
// RUN: %clangxx -c -fsycl -fsycl-targets=intel_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_INPUT
// RUN: %clang_cl -c -fsycl -fsycl-targets=intel_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_INPUT
// BAD_INPUT: error: SYCL target is invalid: 'intel_gpu_bad'

/// test for invalid nvidia arch
// RUN: %clangxx -c -fsycl -fsycl-targets=nvidia_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_NVIDIA_INPUT
// RUN: %clang_cl -c -fsycl -fsycl-targets=nvidia_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_NVIDIA_INPUT
// BAD_NVIDIA_INPUT: error: SYCL target is invalid: 'nvidia_gpu_bad'

/// test for invalid amd arch
// RUN: %clangxx -c -fsycl -fsycl-targets=amd_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_AMD_INPUT
// RUN: %clang_cl -c -fsycl -fsycl-targets=amd_gpu_bad -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD_AMD_INPUT
// BAD_AMD_INPUT: error: SYCL target is invalid: 'amd_gpu_bad'

/// Test for proper creation of fat object
// RUN: %clangxx -c -fsycl -fsycl-targets=intel_gpu_skl \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=FATO
// FATO: clang-offload-bundler{{.*}} "-type=o"
// FATO: "-targets=sycl-spir64_gen-unknown-unknown-skl,host-x86_64-unknown-linux-gnu"

/// Test for proper creation of fat object
// RUN: %clangxx -c -fsycl -fsycl-targets=nvidia_gpu_sm_50 \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NVIDIA_FATO
// NVIDIA_FATO: clang-offload-bundler{{.*}} "-type=o"
// NVIDIA_FATO: "-targets=sycl-nvptx64-nvidia-cuda-sm_50,host-x86_64-unknown-linux-gnu"

/// Test for proper creation of fat object
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx700 \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=AMD_FATO
// AMD_FATO: clang-offload-bundler{{.*}} "-type=o"
// AMD_FATO: "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx700"

/// Test for proper consumption of fat object
// RUN: touch %t.o
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_skl \
// RUN:   -target x86_64-unknown-linux-gnu -### %t.o 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CONSUME_FAT
// CONSUME_FAT: clang-offload-bundler{{.*}} "-type=o"
// CONSUME_FAT: "-targets=host-x86_64-unknown-linux-gnu,sycl-spir64_gen-unknown-unknown-skl"
// CONSUME_FAT: "-unbundle" "-allow-missing-bundles"

/// Test for proper consumption of fat object
// RUN: touch %t.o
// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_50 \
// RUN:   -target x86_64-unknown-linux-gnu -### %t.o 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NVIDIA_CONSUME_FAT
// NVIDIA_CONSUME_FAT: clang-offload-bundler{{.*}} "-type=o"
// NVIDIA_CONSUME_FAT: "-targets=host-x86_64-unknown-linux-gnu,sycl-nvptx64-nvidia-cuda-sm_50"
// NVIDIA_CONSUME_FAT: "-unbundle" "-allow-missing-bundles"

/// Test for proper consumption of fat object
// RUN: touch %t.o
// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx700 \
// RUN:   -target x86_64-unknown-linux-gnu -### %t.o 2>&1 | \
// RUN:   FileCheck %s --check-prefix=AMD_CONSUME_FAT
// AMD_CONSUME_FAT: clang-offload-bundler{{.*}} "-type=o"
// AMD_CONSUME_FAT: "-targets=host-x86_64-unknown-linux-gnu,sycl-amdgcn-amd-amdhsa-gfx700"
// AMD_CONSUME_FAT: "-unbundle" "-allow-missing-bundles"

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
// CHECK_PHASES: 10: linker, {9}, image, (host-sycl)
// CHECK_PHASES: 11: linker, {5}, ir, (device-sycl, skl)
// CHECK_PHASES: 12: sycl-post-link, {11}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES: 13: file-table-tform, {12}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES: 14: llvm-spirv, {13}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES: 15: backend-compiler, {14}, image, (device-sycl, skl)
// CHECK_PHASES: 16: file-table-tform, {12, 15}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES: 17: clang-offload-wrapper, {16}, object, (device-sycl, skl)
// CHECK_PHASES: 18: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_gen-unknown-unknown:skl)" {17}, image

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
// NVIDIA_CHECK_PHASES: 10: linker, {9}, image, (host-sycl)
// NVIDIA_CHECK_PHASES: 11: linker, {5}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 12: sycl-post-link, {11}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 13: file-table-tform, {12}, ir, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 14: backend, {13}, assembler, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: 15: assembler, {14}, object, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: linker, {14, 15}, cuda-fatbin, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: foreach, {13, 16}, cuda-fatbin, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: file-table-tform, {12, 17}, tempfiletable, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: clang-offload-wrapper, {18}, object, (device-sycl, sm_50)
// NVIDIA_CHECK_PHASES: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {19}, image

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
// AMD_CHECK_PHASES: 10: linker, {9}, image, (host-sycl)
// AMD_CHECK_PHASES: 11: linker, {5}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 12: sycl-post-link, {11}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 13: file-table-tform, {12}, ir, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 14: backend, {13}, assembler, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 15: assembler, {14}, object, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 16: linker, {15}, image, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 17: linker, {16}, hip-fatbin, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 18: foreach, {13, 17}, hip-fatbin, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 19: file-table-tform, {12, 18}, tempfiletable, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 20: clang-offload-wrapper, {19}, object, (device-sycl, gfx700)
// AMD_CHECK_PHASES: 21: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (amdgcn-amd-amdhsa:gfx700)" {20}, image

/// Check that ocloc and macro settings only occur for the expected toolchains
/// when mixing spir64_gen and intel_gpu
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg1,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device skl" \
// RUN:   -fno-sycl-device-lib=all -fno-sycl-instrument-device-code \
// RUN:   -target x86_64-unknown-linux-gnu -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CHECK_TOOLS_MIX
// CHECK_TOOLS_MIX: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// CHECK_TOOLS_MIX: "-D__SYCL_TARGET_INTEL_GPU_DG1__"
// CHECK_TOOLS_MIX: ocloc{{.*}} "-device" "dg1"
// CHECK_TOOLS_MIX: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// CHECK_TOOLS_MIX-NOT: "-D__SYCL_TARGET_INTEL_GPU{{.*}}"
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
// CHECK_PHASES_MIX: 10: linker, {9}, image, (host-sycl)
// CHECK_PHASES_MIX: 11: input, "[[INPUT]]", c++, (device-sycl, skl)
// CHECK_PHASES_MIX: 12: preprocessor, {11}, c++-cpp-output, (device-sycl, skl)
// CHECK_PHASES_MIX: 13: compiler, {12}, ir, (device-sycl, skl)
// CHECK_PHASES_MIX: 14: linker, {13}, ir, (device-sycl, skl)
// CHECK_PHASES_MIX: 15: sycl-post-link, {14}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES_MIX: 16: file-table-tform, {15}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES_MIX: 17: llvm-spirv, {16}, tempfilelist, (device-sycl, skl)
// CHECK_PHASES_MIX: 18: backend-compiler, {17}, image, (device-sycl, skl)
// CHECK_PHASES_MIX: 19: file-table-tform, {15, 18}, tempfiletable, (device-sycl, skl)
// CHECK_PHASES_MIX: 20: clang-offload-wrapper, {19}, object, (device-sycl, skl)
// CHECK_PHASES_MIX: 21: linker, {5}, ir, (device-sycl)
// CHECK_PHASES_MIX: 22: sycl-post-link, {21}, tempfiletable, (device-sycl)
// CHECK_PHASES_MIX: 23: file-table-tform, {22}, tempfilelist, (device-sycl)
// CHECK_PHASES_MIX: 24: llvm-spirv, {23}, tempfilelist, (device-sycl)
// CHECK_PHASES_MIX: 25: backend-compiler, {24}, image, (device-sycl)
// CHECK_PHASES_MIX: 26: file-table-tform, {22, 25}, tempfiletable, (device-sycl)
// CHECK_PHASES_MIX: 27: clang-offload-wrapper, {26}, object, (device-sycl)
// CHECK_PHASES_MIX: 28: offload, "host-sycl (x86_64-unknown-linux-gnu)" {10}, "device-sycl (spir64_gen-unknown-unknown:skl)" {20}, "device-sycl (spir64_gen-unknown-unknown)" {27}, image
