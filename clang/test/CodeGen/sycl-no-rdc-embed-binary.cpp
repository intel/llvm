// Tests for -foffload-include-binary with a SYCL fat binary in offload binary
// form (SYCL -fno-sycl-rdc compile-step device embedding via loadLinkModules).

// Missing file: err_cannot_open_file
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary no-such-file -emit-obj -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FILE %s
// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file'

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
// CHECK-MULTI-DAG: @.sycl_offloading.target.1 = internal unnamed_addr constant [11 x i8] c"spir64_gen\00"
// CHECK-MULTI-DAG: @.sycl_offloading.opts.compile.1 = internal unnamed_addr constant [4 x i8] c"-O2\00"
// CHECK-MULTI-DAG: @.sycl_offloading.opts.link.1 = internal unnamed_addr constant [22 x i8] c"-cl-fast-relaxed-math\00"
// CHECK-MULTI-DAG: @.sycl_offloading.1.data = internal unnamed_addr constant [24 x i8] c"SECOND_SYCL_DEVICE_IMAGE"
// CHECK-MULTI-DAG: @__sycl_offload_entry_name.1 = internal unnamed_addr constant [7 x i8] c"_ZTS1L\00"
// CHECK-MULTI: @.sycl_offloading.device_images = internal unnamed_addr constant [2 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 3, i8 4, i8 0, ptr @.sycl_offloading.target.0, ptr @.sycl_offloading.opts.compile.0, ptr @.sycl_offloading.opts.link.0, ptr @.sycl_offloading.0.data, {{.*}}ptr @__sycl_offload_entries_arr, {{.*}}}, %__sycl.tgt_device_image { i16 3, i8 4, i8 2, ptr @.sycl_offloading.target.1, ptr @.sycl_offloading.opts.compile.1, ptr @.sycl_offloading.opts.link.1, ptr @.sycl_offloading.1.data, {{.*}}ptr @__sycl_offload_entries_arr.2, {{.*}}}]
// CHECK-MULTI: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 2, ptr @.sycl_offloading.device_images, ptr null, ptr null }
// CHECK-LINK: define internal void @sycl.descriptor_reg()
// CHECK-LINK: call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)

// Multiple images retain their own target, options and kernel entries, and are
// registered together through a single descriptor.
// RUN: echo -n 'SECOND_SYCL_DEVICE_IMAGE' > %t/image2.bin
// RUN: llvm-offload-binary -o %t/multi-image.syclfb \
// RUN:   --image=file=%t/image.bin,kind=sycl,triple=spir64-unknown-unknown,arch=,sycl-target=spir64,sycl-image-format=0,symbols=_ZTS1K,compile-opts=-g,link-opts=-cl-opt \
// RUN:   --image=file=%t/image2.bin,kind=sycl,triple=spir64_gen-unknown-unknown,arch=,sycl-target=spir64_gen,sycl-image-format=2,symbols=_ZTS1L,compile-opts=-O2,link-opts=-cl-fast-relaxed-math
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t/multi-image.syclfb -emit-llvm %t/host.cpp -o - \
// RUN:   | FileCheck --check-prefixes=CHECK-LINK,CHECK-MULTI %s \
// RUN:     --implicit-check-not='.sycl_fatbin' \
// RUN:     --implicit-check-not='.sycl_offloading.binary'

// A missing sycl-target is accepted as an empty target; it does not fall back
// to the offload binary's triple.
// RUN: llvm-offload-binary -o %t/no-target.syclfb \
// RUN:   --image=file=%t/image.bin,kind=sycl,triple=spir64-unknown-unknown,sycl-image-format=0,symbols=_ZTS1K
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t/no-target.syclfb -emit-llvm %t/host.cpp -o - \
// RUN:   | FileCheck --check-prefix=CHECK-NO-TARGET %s
// CHECK-NO-TARGET: @.sycl_offloading.target.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// CHECK-NO-TARGET: @.sycl_offloading.0.data = internal unnamed_addr constant [22 x i8] c"FAKE_SYCL_DEVICE_IMAGE"
// CHECK-NO-TARGET: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 3, i8 4, i8 0, ptr @.sycl_offloading.target.0,
// CHECK-NO-TARGET: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }
// CHECK-NO-TARGET: call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)

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
