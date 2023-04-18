// RUN: %clang -### -target x86_64 -ffp-accuracy=high -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-H %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-L %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=medium -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-M %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low:sin,cos -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FUNC-1 %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low:sin,cos -ffp-accuracy=high:tan -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FUNC-2 %s

// CHECK-H: "-ffp-accuracy-attr=high"
// CHECK-L: "-ffp-accuracy-attr=low"
// CHECK-M: "-ffp-accuracy-attr=medium"
// CHECK-FUNC-1: "-ffp-accuracy-attr=low:sin,cos"
// CHECK-FUNC-2: "-ffp-accuracy-attr=low:sin,cos high:tan"
