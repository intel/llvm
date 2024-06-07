/// Tests for -fsycl-embed-ir

// UNSUPPORTED: system-windows

// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_80 -fsycl-embed-ir -ccc-print-phases %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-NV %s

// CHECK-NV: [[IR:[0-9]+]]: compiler, {4}, ir, (device-sycl, sm_80)
// CHECK-NV: [[IR_OFFLOAD:[0-9]+]]:  offload, "host-sycl (x86_64-unknown-linux-gnu)" {{{.*}}}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {[[IR]]}, c++-cpp-output
// CHECK-NV: [[COMPILER:[0-9]+]]: compiler, {[[IR_OFFLOAD]]}, ir, (host-sycl)
// CHECK-NV: [[BACKEND:[0-9]+]]: backend, {[[COMPILER]]}, assembler, (host-sycl)
// CHECK-NV: [[ASSEMBLER:[0-9]+]]: assembler, {[[BACKEND]]}, object, (host-sycl)
// CHECK-NV: [[POSTLINK:[0-9]+]]: sycl-post-link, {{{.*}}}, ir, (device-sycl, sm_80)
// CHECK-NV: [[WRAP:[0-9]+]]: clang-offload-wrapper, {[[POSTLINK]]}, object, (device-sycl, sm_80)
// CHECK-NV: [[WRAP_OFFLOAD:[0-9]+]]: offload, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {[[WRAP]]}, object
// CHECK-NV: [[FATBIN_OFFLOAD:[0-9]+]]: offload, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {{{.*}}}, object
// CHECK-NV: linker, {[[ASSEMBLER]], [[WRAP_OFFLOAD]], [[FATBIN_OFFLOAD]]}, image, (host-sycl)

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1010 -fsycl-embed-ir -ccc-print-phases %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-AMD %s

// CHECK-AMD: [[IR:[0-9]+]]: compiler, {4}, ir, (device-sycl, gfx1010)
// CHECK-AMD: [[IR_OFFLOAD:[0-9]+]]:  offload, "host-sycl (x86_64-unknown-linux-gnu)" {{{.*}}}, "device-sycl (amdgcn-amd-amdhsa:gfx1010)" {[[IR]]}, c++-cpp-output
// CHECK-AMD: [[COMPILER:[0-9]+]]: compiler, {[[IR_OFFLOAD]]}, ir, (host-sycl)
// CHECK-AMD: [[BACKEND:[0-9]+]]: backend, {[[COMPILER]]}, assembler, (host-sycl)
// CHECK-AMD: [[ASSEMBLER:[0-9]+]]: assembler, {[[BACKEND]]}, object, (host-sycl)
// CHECK-AMD: [[POSTLINK:[0-9]+]]: sycl-post-link, {{{.*}}}, ir, (device-sycl, gfx1010)
// CHECK-AMD: [[WRAP:[0-9]+]]: clang-offload-wrapper, {[[POSTLINK]]}, object, (device-sycl, gfx1010)
// CHECK-AMD: [[WRAP_OFFLOAD:[0-9]+]]: offload, "device-sycl (amdgcn-amd-amdhsa:gfx1010)" {[[WRAP]]}, object
// CHECK-AMD: [[FATBIN_OFFLOAD:[0-9]+]]: offload, "device-sycl (amdgcn-amd-amdhsa:gfx1010)" {{{.*}}}, object
// CHECK-AMD: linker, {[[ASSEMBLER]], [[WRAP_OFFLOAD]], [[FATBIN_OFFLOAD]]}, image, (host-sycl)
