/// Tests for -fsycl-embed-ir

// UNSUPPORTED: system-windows

// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_80 -fsycl-embed-ir -ccc-print-phases %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-NV %s

// CHECK-NV: [[IR:[0-9]+]]: compiler, {4}, ir, (device-sycl, sm_80)
// CHECK-NV: [[POSTLINK:[0-9]+]]: sycl-post-link, {{{.*}}}, ir, (device-sycl, sm_80)
// CHECK-NV: [[WRAP:[0-9]+]]: clang-offload-wrapper, {[[POSTLINK]]}, object, (device-sycl, sm_80)
// CHECK-NV: offload, "host-sycl (x86_64-unknown-linux-gnu)" {{{.*}}}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {[[WRAP]]}, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {{{.*}}}, image

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1010 -fsycl-embed-ir -ccc-print-phases %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-AMD %s

// CHECK-AMD: [[IR:[0-9]+]]: compiler, {4}, ir, (device-sycl, gfx1010)
// CHECK-AMD: [[POSTLINK:[0-9]+]]: sycl-post-link, {{{.*}}}, ir, (device-sycl, gfx1010)
// CHECK-AMD: [[WRAP:[0-9]+]]: clang-offload-wrapper, {[[POSTLINK]]}, object, (device-sycl, gfx1010)
// CHECK-AMD: offload, "host-sycl (x86_64-unknown-linux-gnu)" {{{.*}}}, "device-sycl (amdgcn-amd-amdhsa:gfx1010)" {[[WRAP]]}, "device-sycl (amdgcn-amd-amdhsa:gfx1010)" {{{.*}}}, image
