// Check the per-translation-unit __sycl_registerlib_<id> reference that SYCL
// host code generation emits. The id is the translation unit's
// -fsycl-unique-prefix, which the driver generates.

// With the new offloading model, the offload object is an offload binary.
// RUN: echo 'define void @f() { ret void }' | llvm-as -o %t.dev.bc
// RUN: llvm-offload-binary -o %t.o --image=file=%t.dev.bc,kind=sycl,triple=spir64-unknown-unknown
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fsycl-unique-prefix=uid42 -emit-llvm -fembed-offload-object=%t.o %s -o - | FileCheck %s

// With the old offloading model, the offload object is the device bitcode.
// Without RDC, the reference is only emitted if it has device code, so an
// object without any can be linked by a host linker.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fno-gpu-rdc -fsycl-unique-prefix=uid42 -emit-llvm -fembed-offload-object=%t.dev.bc %s -o - | FileCheck %s
// RUN: echo 'target datalayout = "e"' | llvm-as -o %t.empty.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fno-gpu-rdc -fsycl-unique-prefix=uid42 -emit-llvm -fembed-offload-object=%t.empty.bc %s -o - | FileCheck %s --check-prefix=EMPTY --implicit-check-not=__sycl_registerlib_

// With RDC, the reference is always emitted, as the SYCL device link defines
// the function.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc -fsycl-unique-prefix=uid42 -emit-llvm -fembed-offload-object=%t.empty.bc %s -o - | FileCheck %s

// Nothing is emitted when the module embeds its device binary
// (-fno-sycl-rdc), as it then registers the binary itself.
// RUN: echo -n 'FAKE_SYCL_DEVICE_IMAGE' > %t.bin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -foffload-include-binary %t.bin -fsycl-unique-prefix=uid42 -emit-llvm %s -o - | FileCheck %s --check-prefix=EMBED --implicit-check-not=__sycl_registerlib_

// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__sycl_registerlib_ctor, ptr null }]
// CHECK: declare void @__sycl_registerlib_uid42()
// CHECK: define internal void @__sycl_registerlib_ctor()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__sycl_registerlib_uid42()
// CHECK-NEXT: ret void

// EMBED: @.sycl_offloading.binary

// EMPTY: define{{.*}} @_Z3foov()

void foo() {}
