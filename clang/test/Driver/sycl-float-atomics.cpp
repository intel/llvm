/// Check that float atomics are NOT "emulated" by default:
// RUN:   %clang -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN:   %clang_cl -### -fsycl -fsycl-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT-NOT: "-D__SYCL_EMULATE_FLOAT_ATOMICS__"

/// Check that "-fsycl-emulate-float-atomics" passes the macro to the device FE:
// RUN:   %clang -### -fsycl -fsycl-emulate-float-atomics %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-EMULATE-ATOMICS %s
// RUN:   %clang_cl -### -fsycl -fsycl-emulate-float-atomics %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-EMULATE-ATOMICS %s
// RUN:   %clang -### -fsycl -fsycl-device-only -fsycl-emulate-float-atomics %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-EMULATE-ATOMICS %s
// RUN:   %clang_cl -### -fsycl -fsycl-device-only -fsycl-emulate-float-atomics %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-EMULATE-ATOMICS %s
// CHECK-SYCL-EMULATE-ATOMICS: clang{{.*}} "-fsycl-is-device"{{.*}} "-D__SYCL_EMULATE_FLOAT_ATOMICS__=1"
