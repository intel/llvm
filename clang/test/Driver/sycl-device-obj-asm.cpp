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
// RUN:   %clang -fsycl-device-only -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -fsycl-device-obj=asm -S %s 2>&1 -ccc-print-phases -o - | FileCheck %s --check-prefix=CHECK-PTX
// CHECK-PTX: 0: input, "{{.+\.cpp}}", c++, (device-sycl, sm_50)
// CHECK-PTX: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, sm_50)
// CHECK-PTX: 2: compiler, {1}, ir, (device-sycl, sm_50)
// CHECK-PTX: 3: backend, {2}, assembler, (device-sycl, sm_50)
// CHECK-PTX: 4: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {3}, assembler

/// Check -fsycl-device-obj option when emitting llvm IR.
// RUN:   %clang -fsycl-device-only -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -fsycl-device-obj=llvmir -S %s 2>&1 -ccc-print-phases -o - | FileCheck %s --check-prefix=CHECK-LLVMIR
// CHECK-LLVMIR: 0: input, "{{.+\.cpp}}", c++, (device-sycl, sm_50)
// CHECK-LLVMIR: 1: preprocessor, {0}, c++-cpp-output, (device-sycl, sm_50)
// CHECK-LLVMIR: 2: compiler, {1}, ir, (device-sycl, sm_50)
// CHECK-LLVMIR: 3: offload, "device-sycl (nvptx64-nvidia-cuda:sm_50)" {2}, ir

/// -fsycl-device-obj=asm should always be accompanied by -fsycl-device-only
/// and -S, check that the compiler issues a correct warning message:
// RUN:   %clang -nocudalib -fsycl-device-only -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -fsycl-device-obj=asm %s 2>&1 -o - | FileCheck %s --check-prefix=CHECK-NO-DEV-ONLY-NO-S
// CHECK-NO-DEV-ONLY-NO-S: warning: -fsycl-device-obj=asm flag has an effect only when compiling device code and emitting assembly, make sure both -fsycl-device-only and -S flags are present; will be ignored [-Wunused-command-line-argument]

/// -fsycl-device-obj=asm will finish at generating assembly stage, hence
/// inform users that generating library will not be possible (ignore -c)
// RUN:   %clang -nocudalib -fsycl-device-only -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -fsycl-device-obj=asm %s 2>&1 -fsycl-device-only -S -c -o - | FileCheck %s --check-prefix=CHECK-DASH-C-IGNORE
// CHECK-DASH-C-IGNORE: warning: argument unused during compilation: '-c' [-Wunused-command-line-argument]
