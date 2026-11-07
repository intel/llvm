// REQUIRES: spirv-registered-target, x86-registered-target

// REQUIRES: spirv-to-ir-wrapper, sycl-post-link

/// Verify that no-RDC emits image address/size records in .tgtimg, while RDC
/// embeds device code for linking in .llvm.offloading.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -fno-sycl-use-header -c %s -o %t.nordc.o
// RUN: llvm-readelf -S %t.nordc.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

/// -fsycl-link also forwards --sycl-device-link to the per-TU no-RDC job.
/// Fatbin-only completion must skip the device-link copy branch.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -fno-sycl-use-header -fsycl-link -c %s -o %t.nordc.devlink.o
// RUN: llvm-readelf -S %t.nordc.devlink.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver --no-offloadlib -fgpu-rdc \
// RUN:   -fno-sycl-use-header -c %s -o %t.rdc.o
// RUN: llvm-readelf -S %t.rdc.o \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.tgtimg'

/// The choice does not depend on the object format.
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver --no-offloadlib -fno-gpu-rdc \
// RUN:   -fno-sycl-use-header -c %s -o %t.nordc.obj
// RUN: llvm-readobj --sections %t.nordc.obj \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'

// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver --no-offloadlib -fgpu-rdc \
// RUN:   -fno-sycl-use-header -c %s -o %t.rdc.obj
// RUN: llvm-readobj --sections %t.rdc.obj \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.tgtimg'

// NORDC: .tgtimg
// RDC: .llvm.offloading

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
  func();
}

void f() { kernel<class TestKernel>([]() {}); }
