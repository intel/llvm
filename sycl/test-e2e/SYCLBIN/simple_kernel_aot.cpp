// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19373

// REQUIRES: ocloc

// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.input_bmg_g21_gpu_device_arch.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" -o %t.input_all_gpu_device_archs.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.object_bmg_g21_gpu_device_arch.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.executable_bmg_g21_gpu_device_arch.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.default_bmg_g21_gpu_device_arch.syclbin %s

// RUN: syclbin-dump %t.input_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH
// RUN: syclbin-dump %t.input_all_gpu_device_archs.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-INPUT-ALL-GPU-DEVICE-ARCHS
// RUN: syclbin-dump %t.object_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH
// RUN: syclbin-dump %t.executable_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH
// RUN: syclbin-dump %t.default_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH

// Checks the generated SYCLBIN contents of a simple SYCL free function kernel.

#include <sycl/sycl.hpp>

extern "C" {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
}
}

// CHECK-GENERAL: Global metadata:
// CHECK-GENERAL-NEXT:   SYCLBIN/global metadata:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH:     state: 0
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS:     state: 0
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH:     state: 1
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH:     state: 2
// CHECK-GENERAL: Abstract Module ID: 0
// CHECK-GENERAL-NEXT: Image Kind: o
// CHECK-GENERAL-NEXT: Triple:{{.*}}spir64_gen-unknown
// CHECK-GENERAL-NEXT: Arch:{{.*}}
// CHECK-GENERAL:   Metadata:
// CHECK-GENERAL-NEXT:   SYCL/device requirements:
// CHECK-GENERAL-NEXT:     aspects:
// CHECK-GENERAL-NEXT:   SYCL/kernel names:
// CHECK-GENERAL-NEXT:     __sycl_kernel_TestKernel: 1
// CHECK-GENERAL-NEXT:   SYCL/misc properties:
// CHECK-GENERAL-NEXT:     optLevel: 2
// CHECK-GENERAL-NEXT: Raw bytes: <Binary blob of {{.*}} bytes>
