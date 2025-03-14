///
/// Check that we call into backend assembler, when using `asm` as device
/// object format, namely:
/// `backend, {2}, assembler, (device-sycl, ...)`

// REQUIRES: nvptx-registered-target,amdgpu-registered-target

/// Check -fsycl-device-obj=asm for AMD.
// RUN:   %clang -fsycl-device-only -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx90a -fsycl-device-obj=asm -S %s 2>&1 -ccc-print-phases -o - | FileCheck %s --check-prefix=CHECK-AMD
// CHECK-AMD: 0: input, "{{.+\.cpp}}", c++, (device-sycl, gfx90a)
// CHECK-AMD: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, gfx90a)
// CHECK-AMD: 2: compiler, {1}, ir, (device-sycl, gfx90a)
// CHECK-AMD: 3: backend, {2}, assembler, (device-sycl, gfx90a)
// CHECK-AMD: 4: offload, "device-sycl (amdgcn-amd-amdhsa:gfx90a)" {3}, assembler

/// Check -fsycl-device-obj=asm for Nvidia.
// RUN:   %clang -fsycl-device-only -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda "--cuda-gpu-arch=sm_50" -fsycl-device-obj=asm -S %s 2>&1 -ccc-print-phases -o - | FileCheck %s --check-prefix=CHECK-PTX
// CHECK-PTX: 0: input, "{{.+\.cpp}}", c++, (device-sycl, sm_50)
// CHECK-PTX: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, sm_50)
// CHECK-PTX: 2: compiler, {1}, ir, (device-sycl, sm_50)
// CHECK-PTX: 3: backend, {2}, assembler, (device-sycl, sm_50)
// CHECK-PTX: 4: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {3}, assembler

/// Check -fsycl-device-obj option when emitting llvm IR.
// RUN:   %clang -fsycl-device-only -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda "--cuda-gpu-arch=sm_50" -fsycl-device-obj=llvmir -S %s 2>&1 -ccc-print-phases -o - | FileCheck %s --check-prefix=CHECK-LLVMIR
// CHECK-LLVMIR: 0: input, "{{.+\.cpp}}", c++, (device-sycl, sm_50)
// CHECK-LLVMIR: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, sm_50)
// CHECK-LLVMIR: 2: compiler, {1}, ir, (device-sycl, sm_50)
// CHECK-LLVMIR: 3: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {2}, ir
