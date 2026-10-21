// Tests for -foffload-include-binary with a SYCL wrapper module in bitcode form
// (SYCL -fno-sycl-rdc compile-step device embedding via loadLinkModules).

// Missing file: err_cannot_open_file
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary no-such-file.bc -emit-obj -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FILE %s
// CHECK-NO-FILE: fatal error: cannot open file 'no-such-file.bc'

// Wrapper module bitcode: linked into the host module as is, so its
// registration constructor is kept and no raw device binary is embedded.
// RUN: split-file %s %t
// RUN: llvm-as %t/wrapper.ll -o %t/wrapper.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t/wrapper.bc -emit-llvm %t/host.cpp -o - \
// RUN:   | FileCheck --check-prefix=CHECK-LINK %s \
// RUN:     --implicit-check-not='.sycl_fatbin' \
// RUN:     --implicit-check-not='.sycl_offloading.binary'
// CHECK-LINK: @llvm.global_ctors = {{.*}} ptr @sycl.descriptor_reg
// CHECK-LINK: define internal void @sycl.descriptor_reg()
// CHECK-LINK: call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)

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

//--- wrapper.ll
@.sycl_offloading.descriptor = internal constant [1 x i8] zeroinitializer
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @sycl.descriptor_reg, ptr null }]

define internal void @sycl.descriptor_reg() {
  call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
  ret void
}

declare void @__sycl_register_lib(ptr)

//--- host.cpp
void f() {}

//--- kernel.cu
__attribute__((global)) void k() {}
