/// Tests the behaviors of using -fsycl --offload-new-driver 
//  --offload-arch=<intel-gpu-values>.

// SYCL AOT compilation to Intel GPUs using --offload-arch

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=bdw %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=bdw -DMAC_STR=BDW

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=skl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=skl -DMAC_STR=SKL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=kbl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=kbl -DMAC_STR=KBL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=cfl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=cfl -DMAC_STR=CFL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=apl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=apl -DMAC_STR=APL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=bxt %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=apl -DMAC_STR=APL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=glk %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=glk -DMAC_STR=GLK

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=whl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=whl -DMAC_STR=WHL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=aml %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=aml -DMAC_STR=AML

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=cml %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=cml -DMAC_STR=CML

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=icllp %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=icllp -DMAC_STR=ICLLP

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=icl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=icllp -DMAC_STR=ICLLP

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=ehl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=ehl -DMAC_STR=EHL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=jsl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=ehl -DMAC_STR=EHL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=tgllp %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=tgllp -DMAC_STR=TGLLP

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=tgl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=tgllp -DMAC_STR=TGLLP

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=rkl %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=rkl -DMAC_STR=RKL

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=adl_s %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=adl_s -DMAC_STR=ADL_S

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=rpl_s %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=adl_s -DMAC_STR=ADL_S

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=adl_p %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=adl_p -DMAC_STR=ADL_P

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=adl_n %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=adl_n -DMAC_STR=ADL_N

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=dg1 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=dg1 -DMAC_STR=DG1

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=acm_g10 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=acm_g10 -DMAC_STR=ACM_G10

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=dg2_g10 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=acm_g10 -DMAC_STR=ACM_G10

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=acm_g11 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=acm_g11 -DMAC_STR=ACM_G11

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=dg2_g11 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=acm_g11 -DMAC_STR=ACM_G11

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=acm_g12 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=acm_g12 -DMAC_STR=ACM_G12

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=dg2_g12 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU -DDEV_STR=acm_g12 -DMAC_STR=ACM_G12

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=pvc %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=pvc -DMAC_STR=PVC

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=pvc_vg %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=pvc_vg -DMAC_STR=PVC_VG

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=mtl_u %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=mtl_u -DMAC_STR=MTL_U

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=mtl_s %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=mtl_u -DMAC_STR=MTL_U

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=arl_u %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=mtl_u -DMAC_STR=MTL_U

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=arl_s %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=mtl_u -DMAC_STR=MTL_U

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=mtl_h %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=mtl_h -DMAC_STR=MTL_H

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=arl_h %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=arl_h -DMAC_STR=ARL_H

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=bmg_g21 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=bmg_g21 -DMAC_STR=BMG_G21

// RUN: %clangxx -### --offload-new-driver -fsycl --offload-arch=lnl_m %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=TARGET-TRIPLE-GPU,CLANG-OFFLOAD-PACKAGER-GPU-OPTS -DDEV_STR=lnl_m -DMAC_STR=LNL_M

// TARGET-TRIPLE-GPU: clang{{.*}} "-triple" "spir64_gen-unknown-unknown"
// TARGET-TRIPLE-GPU: "-D__SYCL_TARGET_INTEL_GPU_[[MAC_STR]]__"
// CLANG-OFFLOAD-PACKAGER-GPU: clang-offload-packager{{.*}} "--image={{.*}}triple=spir64_gen-unknown-unknown,arch=[[DEV_STR]],kind=sycl"
// CLANG-OFFLOAD-PACKAGER-GPU-OPTS: clang-offload-packager{{.*}} "--image={{.*}}triple=spir64_gen-unknown-unknown,arch=[[DEV_STR]],kind=sycl{{.*}}"

