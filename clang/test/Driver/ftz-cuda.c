// RUN: %clang -### -fcuda-flush-denormals-to-zero -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FTZ %s
// CHECK-FTZ: "-cc1"
// CHECK-FTZ: "-fdenormal-fp-math-f32=preserve-sign,preserve-sign"
