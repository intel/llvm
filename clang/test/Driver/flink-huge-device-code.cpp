// UNSUPPORTED: system-windows
// Test the addition of a custom linker script for huge device code.

// RUN: %clangxx -### -fsycl -flink-huge-device-code %s 2>&1 | \
// RUN:       FileCheck --check-prefix=CHECK-LINKER-SCRIPT %s
// RUN: %clangxx -### -fopenmp -fopenmp-targets=x86_64 -flink-huge-device-code %s 2>&1 | \
// RUN:       FileCheck --check-prefix=CHECK-LINKER-SCRIPT %s
// CHECK-LINKER-SCRIPT: "-T" "{{.*}}.ld"

// Also check that a user-provided linker script may be used:
// RUN: %clangxx -### -fsycl -flink-huge-device-code %s \
// RUN:       -T custom-user-script.ld 2>&1 | \
// RUN:       FileCheck --check-prefixes=CHECK-USER-SCRIPT %s
// RUN: %clangxx -### -fopenmp -fopenmp-targets=x86_64 -flink-huge-device-code %s \
// RUN:       -T custom-user-script.ld 2>&1 | \
// RUN:       FileCheck --check-prefixes=CHECK-USER-SCRIPT %s
// CHECK-USER-SCRIPT: "-T" "custom-user-script.ld"
