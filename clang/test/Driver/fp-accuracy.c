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

// RUN: not %clang -Xclang -verify -ffp-accuracy=foo %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=CHECK-ERR

// RUN: not %clang -Xclang -verify -ffp-accuracy=foo:[sin,cos] %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=CHECK-ERR

// RUN: not %clang -Xclang -verify -ffp-accuracy=foo:[sin,cos] \
// RUN: -ffp-accuracy=goo %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=CHECK-ERR

// RUN: not %clang -Xclang -verify -ffp-accuracy=foo:[sin,cos] \
// RUN: -ffp-accuracy=goo:[tan] %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=CHECK-ERR-1

// RUN: not %clang -Xclang -verify -ffp-accuracy=high=[sin] %s 2>& 1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-ERR-2

// RUN: not %clang -Xclang -verify -ffp-accuracy=low:[sin,cos] \
// RUN: -ffp-accuracy=high %s 2>&1  \
// RUN: | FileCheck %s --check-prefix=WARN

// CHECK-H: "-ffp-builtin-accuracy=high"
// CHECK-L: "-ffp-builtin-accuracy=low"
// CHECK-M: "-ffp-builtin-accuracy=medium"
// CHECK-FUNC-1: "-ffp-builtin-accuracy=low:sin,cos"
// CHECK-FUNC-2: "-ffp-builtin-accuracy=low:sin,cos high:tan"
// CHECK-ERR: (frontend): unsupported argument 'foo' to option 'ffp-accuracy'
// CHECK-ERR-1: (frontend): unsupported argument 'foo' to option 'ffp-accuracy'
// CHECK-ERR-1: (frontend): unsupported argument 'foo' to option 'ffp-accuracy'
// CHECK-ERR-1: (frontend): unsupported argument 'goo' to option 'ffp-accuracy'
// CHECK-ERR-2: (frontend): unsupported argument 'high=[sin]' to option 'ffp-accuracy'
// WARN: (frontend): FP accuracy value of 'high' has already been assigned to function 'cos'
// WARN: (frontend): FP accuracy value of 'high' has already been assigned to function 'sin'
