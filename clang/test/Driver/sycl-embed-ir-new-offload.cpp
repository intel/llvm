/// Tests for -fsycl-embed-ir

// UNSUPPORTED: system-windows

// RUN: %clangxx -fsycl -fsycl-targets=nvidia_gpu_sm_80 --offload-new-driver -fsycl-embed-ir -ccc-print-phases %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-NV %s

// CHECK-NV: [[ZERO:[0-9]+]]: input, {{.*}}, c++, (host-sycl)
// CHECK-NV: [[ONE:[0-9]+]]: append-footer, {[[ZERO]]}, c++, (host-sycl)
// CHECK-NV: [[TWO:[0-9]+]]: preprocessor, {[[ONE]]}, c++-cpp-output, (host-sycl)
// CHECK-NV: [[THREE:[0-9]+]]: compiler, {[[TWO]]}, ir, (host-sycl)
// CHECK-NV: [[FOUR:[0-9]+]]: input, {{.*}}, c++, (device-sycl, sm_80)
// CHECK-NV: [[FIVE:[0-9]+]]: preprocessor, {[[FOUR]]}, c++-cpp-output, (device-sycl, sm_80)
// CHECK-NV: [[SIX:[0-9]+]]: compiler, {[[FIVE]]}, ir, (device-sycl, sm_80)
// CHECK-NV: [[SEVEN:[0-9]+]]: backend, {[[SIX]]}, ir, (device-sycl, sm_80)
// CHECK-NV: [[EIGHT:[0-9]+]]:  offload, "device-sycl (nvptx64-nvidia-cuda:sm_80)" {[[SEVEN]]}, ir
// CHECK-NV: [[NINE:[0-9]+]]:  clang-offload-packager, {[[EIGHT]]}, image, (device-sycl)
// CHECK-NV: [[TEN:[0-9]+]]:  offload, "host-sycl (x86_64-unknown-linux-gnu)" {[[THREE]]}, "device-sycl (x86_64-unknown-linux-gnu)" {[[NINE]]}, ir
// CHECK-NV: [[ELEVEN:[0-9]+]]:  backend, {[[TEN]]}, assembler, (host-sycl)
// CHECK-NV: [[TWELVE:[0-9]+]]:  assembler, {[[ELEVEN]]}, object, (host-sycl)
// CHECK-NV: [[LAST:[0-9]+]]:  clang-linker-wrapper, {[[TWELVE]]}, image, (host-sycl)

// RUN: %clangxx -fsycl -fsycl-targets=amd_gpu_gfx1010 -fsycl-embed-ir --offload-new-driver -ccc-print-phases %s 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-AMD %s

// CHECK-AMD: [[ZERO:[0-9]+]]: input, {{.*}}, c++, (host-sycl)
// CHECK-AMD: [[ONE:[0-9]+]]: append-footer, {[[ZERO]]}, c++, (host-sycl)
// CHECK-AMD: [[TWO:[0-9]+]]: preprocessor, {[[ONE]]}, c++-cpp-output, (host-sycl)
// CHECK-AMD: [[THREE:[0-9]+]]: compiler, {[[TWO]]}, ir, (host-sycl)
// CHECK-AMD: [[FOUR:[0-9]+]]: input, {{.*}}, c++, (device-sycl, gfx1010)
// CHECK-AMD: [[FIVE:[0-9]+]]: preprocessor, {[[FOUR]]}, c++-cpp-output, (device-sycl, gfx1010)
// CHECK-AMD: [[SIX:[0-9]+]]: compiler, {[[FIVE]]}, ir, (device-sycl, gfx1010)
// CHECK-AMD: [[SEVEN:[0-9]+]]: backend, {[[SIX]]}, ir, (device-sycl, gfx1010)
// CHECK-AMD: [[EIGHT:[0-9]+]]:  offload, "device-sycl (amdgcn-amd-amdhsa:gfx1010)" {[[SEVEN]]}, ir
// CHECK-AMD: [[NINE:[0-9]+]]:  clang-offload-packager, {[[EIGHT]]}, image, (device-sycl)
// CHECK-AMD: [[TEN:[0-9]+]]:  offload, "host-sycl (x86_64-unknown-linux-gnu)" {[[THREE]]}, "device-sycl (x86_64-unknown-linux-gnu)" {[[NINE]]}, ir
// CHECK-AMD: [[ELEVEN:[0-9]+]]:  backend, {[[TEN]]}, assembler, (host-sycl)
// CHECK-AMD: [[TWELVE:[0-9]+]]:  assembler, {[[ELEVEN]]}, object, (host-sycl)
// CHECK-AMD: [[LAST:[0-9]+]]:  clang-linker-wrapper, {[[TWELVE]]}, image, (host-sycl)
