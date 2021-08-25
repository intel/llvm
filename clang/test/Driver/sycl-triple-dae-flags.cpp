// RUN: %clangxx -### -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-dead-args-optimization %s 2> %t.cuda.out
// RUN: FileCheck %s --input-file %t.cuda.out
//
// RUN: %clangxx -### -fsycl -fsycl-targets=amdgcn-amd-amdhsa-sycldevice -Xsycl-target-backend --offload-arch=gfx906 -fsycl-dead-args-optimization %s 2> %t.rocm.out
// RUN: FileCheck %s --input-file %t.rocm.out
// CHECK-NOT: -fenable-sycl-dae
// CHECK-NOT: -emit-param-info
//
// RUN: %clangxx -### -fsycl -fsycl-targets=spir64-unknown-unknown-sycldevice -fsycl-dead-args-optimization %s 2> %t.out
// RUN: FileCheck %s --check-prefixes=CHECK-FENABLE,CHECK-EMIT --input-file %t.out
// CHECK-FENABLE: -fenable-sycl-dae
// CHECK-EMIT: -emit-param-info
