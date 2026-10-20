// REQUIRES: spirv-registered-target, x86-registered-target

// REQUIRES: spirv-to-ir-wrapper, sycl-post-link

/// Verify which section carries the SYCL device binary in the host object.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -c %s -o %t.nordc.o
// RUN: llvm-readelf -S %t.nordc.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

/// -fsycl-link also forwards --sycl-device-link to the per-TU no-RDC job.
/// Fatbin-only completion must skip the device-link copy branch.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -fsycl-link -c %s -o %t.nordc.devlink.o
// RUN: llvm-readelf -S %t.nordc.devlink.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fgpu-rdc \
// RUN:   -c %s -o %t.rdc.o
// RUN: llvm-readelf -S %t.rdc.o \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.tgtimg'

/// The choice does not depend on the object format.
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -c %s -o %t.nordc.obj
// RUN: llvm-readobj --sections %t.nordc.obj \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver --no-offloadlib -fgpu-rdc \
// RUN:   -c %s -o %t.rdc.obj
// RUN: llvm-readobj --sections %t.rdc.obj \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.tgtimg'

// NORDC: .tgtimg
// RDC: .llvm.offloading

/// The host compiler wraps the fat binary into the runtime's descriptor-based
/// registration ABI rather than registering the raw container.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -S -emit-llvm %s -o - | FileCheck -check-prefix=WRAPPER %s

// WRAPPER: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc
// WRAPPER: call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
// WRAPPER-NOT: call void @__sycl_register_lib(ptr {{.*}}, i64

void f() {}
