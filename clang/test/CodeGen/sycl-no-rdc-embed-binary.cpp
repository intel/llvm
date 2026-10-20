// Tests for -foffload-include-binary with a SYCL fat binary in offload binary
// form (SYCL -fno-sycl-rdc compile-step device embedding via loadLinkModules).

// Missing file: err_cannot_open_file
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary no-such-file.bc -emit-obj -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FILE %s
// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file.bc'

// Offload binary: each image is wrapped into the descriptor the SYCL runtime
// registers, together with its target, options and kernel entries. The raw
// container itself is not embedded.
// RUN: split-file %s %t
// RUN: echo -n 'FAKE_SYCL_DEVICE_IMAGE' > %t/image.bin
// RUN: llvm-offload-binary -o %t/fatbin.syclfb \
// RUN:   --image=file=%t/image.bin,kind=sycl,triple=spir64-unknown-unknown,arch=,sycl-target=spir64,sycl-image-format=0,symbols=_ZTS1K,compile-opts=-g,link-opts=-cl-opt
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t/fatbin.syclfb -emit-llvm %t/host.cpp -o - \
// RUN:   | FileCheck --check-prefix=CHECK-LINK %s \
// RUN:     --implicit-check-not='.sycl_fatbin' \
// RUN:     --implicit-check-not='.sycl_offloading.binary'
// CHECK-LINK-DAG: @.sycl_offloading.target.0 = internal unnamed_addr constant [7 x i8] c"spir64\00"
// CHECK-LINK-DAG: @.sycl_offloading.opts.compile.0 = internal unnamed_addr constant [3 x i8] c"-g\00"
// CHECK-LINK-DAG: @.sycl_offloading.opts.link.0 = internal unnamed_addr constant [8 x i8] c"-cl-opt\00"
// CHECK-LINK-DAG: @.sycl_offloading.0.data = internal unnamed_addr constant [22 x i8] c"FAKE_SYCL_DEVICE_IMAGE"
// CHECK-LINK-DAG: @__sycl_offload_entry_name = internal unnamed_addr constant [7 x i8] c"_ZTS1K\00"
// CHECK-LINK-DAG: @llvm.global_ctors = {{.*}} ptr @sycl.descriptor_reg
// CHECK-LINK: define internal void @sycl.descriptor_reg()
// CHECK-LINK: call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)

// Malformed runtime metadata is diagnosed.
// RUN: llvm-offload-binary -o %t/no-format.syclfb \
// RUN:   --image=file=%t/image.bin,kind=sycl,triple=spir64-unknown-unknown
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t/no-format.syclfb -emit-llvm %t/host.cpp \
// RUN:   -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NO-FORMAT %s
// CHECK-NO-FORMAT: error: cannot link module '{{.*}}no-format.syclfb': invalid SYCL image format ''

// RUN: llvm-offload-binary -o %t/openmp.syclfb \
// RUN:   --image=file=%t/image.bin,kind=openmp,triple=spir64-unknown-unknown
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t/openmp.syclfb -emit-llvm %t/host.cpp \
// RUN:   -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-NOT-SYCL %s
// CHECK-NOT-SYCL: error: cannot link module '{{.*}}openmp.syclfb': offload binary entry 0 is not a SYCL device image

// SYCL CUDA compatibility mode: the host compilation is also a CUDA host
// compilation and the file is the CUDA fat binary. Only CUDA registers it; it
// is neither linked nor embedded as a SYCL device image.
// RUN: echo -n 'FAKE_CUDA_FATBIN' > %t/fake.fatbin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda \
// RUN:   -fsycl-is-host -fsycl-cuda-compatibility -fno-gpu-rdc -x cuda \
// RUN:   -foffload-include-binary %t/fake.fatbin -emit-llvm %t/kernel.cu -o - \
// RUN:   | FileCheck --check-prefix=CHECK-SYCL-CUDA %s \
// RUN:     --implicit-check-not='.sycl_fatbin' \
// RUN:     --implicit-check-not='__sycl_register_lib'
// CHECK-SYCL-CUDA: section ".nv_fatbin"
// CHECK-SYCL-CUDA: call ptr @__cudaRegisterFatBinary(

//--- host.cpp
void f() {}

//--- kernel.cu
__attribute__((global)) void k() {}
