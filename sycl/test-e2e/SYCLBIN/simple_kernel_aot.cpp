// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/19373

// REQUIRES: ocloc

// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.input_bmg_g21_gpu_device_arch.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=input -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" -o %t.input_all_gpu_device_archs.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=object -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.object_bmg_g21_gpu_device_arch.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.executable_bmg_g21_gpu_device_arch.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g21" -o %t.default_bmg_g21_gpu_device_arch.syclbin %s

// RUN: syclbin-dump %t.input_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefix CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH
// RUN: syclbin-dump %t.input_all_gpu_device_archs.syclbin | FileCheck %s --check-prefix CHECK-INPUT-ALL-GPU-DEVICE-ARCHS
// RUN: syclbin-dump %t.object_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefix CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH
// RUN: syclbin-dump %t.executable_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefix CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH
// RUN: syclbin-dump %t.default_bmg_g21_gpu_device_arch.syclbin | FileCheck %s --check-prefix CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH

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

// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH:      Version: {{[1-9]+}}
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Global metadata:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   SYCLBIN/global metadata:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:     state: 0
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Number of Abstract Modules: 1
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Abstract Module 0:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Metadata:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH:      Number of IR Modules: 0
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Number of Native Device Code Images: 1
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Native device code image 0:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Metadata:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:     SYCLBIN/native device code image metadata:
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:        arch:{{.*}}
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:        target:{{.*}}spir64_gen-unknown
// CHECK-INPUT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Raw native device code image bytes: <Binary blob of {{.*}} bytes>

// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS:      Version: {{[1-9]+}}
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT: Global metadata:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:   SYCLBIN/global metadata:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:     state: 0
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT: Number of Abstract Modules: 1
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT: Abstract Module 0:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:   Metadata:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS:      Number of IR Modules: 0
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT: Number of Native Device Code Images: 1
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT: Native device code image 0:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:   Metadata:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:     SYCLBIN/native device code image metadata:
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:        arch:{{.*}}
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:        target:{{.*}}spir64_gen-unknown
// CHECK-INPUT-ALL-GPU-DEVICE-ARCHS-NEXT:   Raw native device code image bytes: <Binary blob of {{.*}} bytes>

// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH:      Version: {{[1-9]+}}
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Global metadata:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   SYCLBIN/global metadata:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:     state: 1
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Number of Abstract Modules: 1
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Abstract Module 0:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Metadata:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH:      Number of IR Modules: 0
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Number of Native Device Code Images: 1
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT: Native device code image 0:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Metadata:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:     SYCLBIN/native device code image metadata:
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:        arch:{{.*}}
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:        target:{{.*}}spir64_gen-unknown
// CHECK-OBJECT-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Raw native device code image bytes: <Binary blob of {{.*}} bytes>

// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH:      Version: {{[1-9]+}}
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT: Global metadata:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:   SYCLBIN/global metadata:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:     state: 2
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT: Number of Abstract Modules: 1
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT: Abstract Module 0:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Metadata:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH:      Number of IR Modules: 0
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT: Number of Native Device Code Images: 1
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT: Native device code image 0:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Metadata:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:     SYCLBIN/native device code image metadata:
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:        arch:{{.*}}
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:        target:{{.*}}spir64_gen-unknown
// CHECK-EXECUTABLE-BMG-G21-GPU-DEVICE-ARCH-NEXT:   Raw native device code image bytes: <Binary blob of {{.*}} bytes>
