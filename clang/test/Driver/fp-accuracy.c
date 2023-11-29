// RUN: %clang -### -target x86_64 -ffp-accuracy=high -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=HIGH %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=LOW %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=medium -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=MEDIUM %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=sycl -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SYCL %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=cuda -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CUDA %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low:sin,cos -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FUNC-1 %s

// RUN: %clang -### -target x86_64 -ffp-accuracy=low:sin,cos -ffp-accuracy=high:tan -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FUNC-2 %s

// RUN: not %clang -Xclang -verify -fno-math-errno -ffp-accuracy=foo %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=ERR

// RUN: not %clang -Xclang -verify -fno-math-errno -ffp-accuracy=foo:[sin,cos] %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=ERR

// RUN: not %clang -Xclang -verify -fno-math-errno -ffp-accuracy=foo:[sin,cos] \
// RUN: -ffp-accuracy=goo %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=ERR

// RUN: not %clang -Xclang -verify -fno-math-errno -ffp-accuracy=foo:[sin,cos] \
// RUN: -ffp-accuracy=goo:[tan] %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=ERR-1

// RUN: not %clang -Xclang -verify -fno-math-errno -ffp-accuracy=high=[sin] %s 2>& 1 \
// RUN: | FileCheck %s --check-prefixes=ERR-2

// RUN: not %clang -fno-math-errno -ffp-accuracy=low:[sin,cos] \
// RUN: -ffp-accuracy=high %s 2>&1  \
// RUN: | FileCheck %s --check-prefix=WARN1

// RUN: not %clang -fno-math-errno -ffp-accuracy=low:[sin,cos] \
// RUN: -ffp-accuracy=high:[cos,tan] %s 2>&1  \
// RUN: | FileCheck %s --check-prefix=WARN2

// RUN: not %clang -Xclang -verify -ffp-accuracy=low:[sin,cos] \
// RUN: -ffp-accuracy=high -fmath-errno %s 2>&1  \
// RUN: | FileCheck %s --check-prefix=ERR-3

// RUN: not %clang -Xclang -verify -ffp-accuracy=high \
// RUN: -fmath-errno %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=ERR-3

// HIGH: "-ffp-builtin-accuracy=high"
// LOW: "-ffp-builtin-accuracy=low"
// MEDIUM: "-ffp-builtin-accuracy=medium"
// SYCL: "-ffp-builtin-accuracy=sycl"
// CUDA: "-ffp-builtin-accuracy=cuda"
// FUNC-1: "-ffp-builtin-accuracy=low:sin,cos"
// FUNC-2: "-ffp-builtin-accuracy=low:sin,cos high:tan"
// ERR: (frontend): unsupported argument 'foo' to option '-ffp-accuracy'
// ERR-1: (frontend): unsupported argument 'foo' to option '-ffp-accuracy'
// ERR-2: (frontend): unsupported argument 'high=[sin]' to option '-ffp-accuracy'
// WARN1: '-ffp-accuracy=high' overrides '-ffp-accuracy=low:[sin,cos]' for the function 'cos'
// WARN1: '-ffp-accuracy=high' overrides '-ffp-accuracy=low:[sin,cos]' for the function 'sin'
// WARN2: '-ffp-accuracy=high:[cos,tan]' overrides '-ffp-accuracy=low:[sin,cos]' for the function 'cos'


// ERR-3: (frontend): floating point accuracy requirements cannot be guaranteed when '-fmath-errno' is enabled; use '-fno-math-errno' to enable floating point accuracy control
