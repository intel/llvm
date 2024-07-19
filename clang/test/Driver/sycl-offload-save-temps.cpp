/// SYCL offloading tests using -save-temps

/// Verify that -save-temps does not crash
// RUN: %clang -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-SAVE-TEMPS
// RUN: %clang -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-SAVE-TEMPS
// RUN: %clangxx -fsycl --offload-new-driver -fno-sycl-instrument-device-code -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHK-FSYCL-SAVE-TEMPS
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-fsycl-is-device"{{.*}} "-o" "[[DEVICE_BASE_NAME:[a-z0-9-]+]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-fsycl-is-device"{{.*}} "-fsycl-int-header=[[HEADER_NAME:.+\-header.+\.h]]" "-fsycl-int-footer={{.*}}"{{.*}} "-o" "[[DEVICE_BASE_NAME]].bc"{{.*}} "[[DEVICE_BASE_NAME]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang-offload-packager{{.*}}
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-include" "[[HEADER_NAME]]"{{.*}} "-fsycl-is-host"{{.*}} "-o" "[[HOST_BASE_NAME:[a-z0-9_-]+]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-o" "[[HOST_BASE_NAME:.*]].bc"{{.*}} "[[HOST_BASE_NAME:[a-z0-9_-]+]].ii"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-o" "[[HOST_BASE_NAME:.*]].s"{{.*}} "[[HOST_BASE_NAME]].bc"
// CHK-FSYCL-SAVE-TEMPS: clang{{.*}} "-o" "[[HOST_BASE_NAME:.*]].o"{{.*}} "[[HOST_BASE_NAME]].s"
// CHK-FSYCL-SAVE-TEMPS: ld{{.*}} "[[HOST_BASE_NAME]].o"

/// Verify that -save-temps puts header/footer in a correct place
// RUN: %clang -fsycl --offload-new-driver -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 | FileCheck %s -check-prefixes=CHECK-SAVE-TEMPS-DIR
// CHECK-SAVE-TEMPS-DIR: clang{{.*}} "-fsycl-int-header=sycl-offload-save-temps-header-{{[a-z0-9]*}}.h"{{.*}}"-fsycl-int-footer=sycl-offload-save-temps-footer-{{[a-z0-9]*}}.h"

/// Verify that -save-temps=obj respects the -o dir
// RUN: %clang -fsycl --offload-new-driver -fno-sycl-device-lib=all -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-temps=obj -o %S %s -### 2>&1 | FileCheck %s -check-prefixes=CHECK-SAVE-TEMPS-OBJ-DIR
// CHECK-SAVE-TEMPS-OBJ-DIR: clang{{.*}}-fsycl-int-header={{.*[/\\]+clang[/\\]+test[/\\]+sycl-offload-save-temps-header-[a-z0-9]*}}.h{{.*}}-fsycl-int-footer={{.*[/\\]+clang[/\\]+test[/\\]+sycl-offload-save-temps-footer-[a-z0-9]*}}.h

/// Usage of -save-temps should not set -disable-llvm-passes for device
// RUN: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK_LLVM_PASSES
// RUN: %clang -fsycl --offload-new-driver -fsycl-targets=spir64-unknown-unknown -target x86_64-unknown-linux-gnu -save-temps %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHK_LLVM_PASSES
// CHK_LLVM_PASSES-NOT: clang{{.*}} "-triple" "spir64-unknown-unknown" {{.*}} "-disable-llvm-passes"
// CHK_LLVM_PASSES: clang{{.*}} "-triple" "x86_64-unknown-linux-gnu" {{.*}} "-disable-llvm-passes"
