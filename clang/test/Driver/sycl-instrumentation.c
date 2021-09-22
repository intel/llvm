/// Check if -fsycl-instrument-device-code is passed to device-side -cc1:
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -### %s 2>&1 \
// RUN: | FileCheck -check-prefixes=CHECK-SPIRV,CHECK-HOST %s
// RUN: %clangxx -fsycl -fsycl-targets=nvptx-nvidia-cuda -### %s 2>&1 \
// RUN: | FileCheck -check-prefix=CHECK-NONSPIRV %s
// CHECK-SPIRV: "-cc1"{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-instrument-device-code"
// CHECK-HOST-NOT: "-cc1"{{.*}} "-fsycl-is-host"{{.*}} "-fsycl-instrument-device-code"
// CHECK-NONSPIRV-NOT: "-fsycl-instrument-device-code"
