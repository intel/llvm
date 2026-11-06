// REQUIRES: sycl-post-link

/// Verify that the no-RDC driver flow registers and unregisters the descriptor
/// expected by the Intel SYCL runtime.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-gpu-rdc -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-gpu-rdc -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s

// CHECK: call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
// CHECK: call void @__sycl_unregister_lib(ptr @.sycl_offloading.descriptor)

void f() {}
