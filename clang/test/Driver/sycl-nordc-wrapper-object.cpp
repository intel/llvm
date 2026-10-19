// REQUIRES: spirv-registered-target, x86-registered-target
// REQUIRES: spirv-to-ir-wrapper, sycl-post-link

/// intel/llvm counterpart of LLORG's sycl-nordc-fatbin.cpp. In intel/llvm the
/// non-RDC host object carries the wrapper module produced at compile time:
/// the device image in ".tgtimg" plus the __sycl_register_lib registration,
/// and no ".llvm.offloading", so the final link does not device-link it again.
/// The RDC host object is the reverse.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-use-header -fno-gpu-rdc -c %s -o %t.nordc.o
// RUN: llvm-readelf -S %t.nordc.o \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'
// RUN: llvm-nm %t.nordc.o | FileCheck -check-prefix=REG %s

// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-use-header -fgpu-rdc -c %s -o %t.rdc.o
// RUN: llvm-readelf -S %t.rdc.o \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.tgtimg'
// RUN: llvm-nm %t.rdc.o | FileCheck -check-prefix=NOREG %s

/// The choice does not depend on the object format.
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-use-header -fno-gpu-rdc -c %s -o %t.nordc.obj
// RUN: llvm-readobj --sections %t.nordc.obj \
// RUN:   | FileCheck -check-prefix=NORDC %s --implicit-check-not='.llvm.offloading'
// RUN: llvm-nm %t.nordc.obj | FileCheck -check-prefix=REG %s

// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-use-header -fgpu-rdc -c %s -o %t.rdc.obj
// RUN: llvm-readobj --sections %t.rdc.obj \
// RUN:   | FileCheck -check-prefix=RDC %s --implicit-check-not='.tgtimg'
// RUN: llvm-nm %t.rdc.obj | FileCheck -check-prefix=NOREG %s

// Empty device modules need neither a device image nor runtime registration.
// RUN: %clangxx --target=x86_64-unknown-linux-gnu -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-use-header -fno-gpu-rdc -DEMPTY -c %s -o %t.empty.o
// RUN: llvm-readobj --sections --symbols %t.empty.o | FileCheck -check-prefix=EMPTY %s
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:   --no-offloadlib -fno-sycl-use-header -fno-gpu-rdc -DEMPTY -c %s -o %t.empty.obj
// RUN: llvm-readobj --sections --symbols %t.empty.obj | FileCheck -check-prefix=EMPTY %s

// NORDC: .tgtimg
// RDC: .llvm.offloading
// REG: U __sycl_register_lib
// NOREG-NOT: __sycl_register_lib
// EMPTY: Format:
// EMPTY-NOT: .tgtimg
// EMPTY-NOT: .llvm.offloading
// EMPTY-NOT: __sycl_register_lib
// EMPTY-NOT: __sycl_unregister_lib

#ifndef EMPTY
template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &func) {
  func();
}

void f() { kernel<class TestKernel>([]() {}); }
#else
void f() {}
#endif
