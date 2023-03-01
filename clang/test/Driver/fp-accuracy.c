// TODO: add a check line for all values of fp-accuracy.

// RUN: %clang -### -target x86_64 -ffp-accuracy=high -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low:sin,cos -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FUNC %s

// CHECK: "-ffp-accuracy=high"
// CHECK-FUNC: "-fpbuiltin-max-error=low:sin,cos"

