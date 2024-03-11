// RUN: %clang -target i386-apple-darwin11 -m32 -Xarch_i386 -O5 %s -S -### 2>&1 | FileCheck -check-prefix=O5ONCE %s
// O5ONCE: "-O5"
// O5ONCE-NOT: "-O5"

// RUN: %clang -target i386-apple-darwin11 -m64 -Xarch_i386 -O5 %s -S -### 2>&1 | FileCheck -check-prefix=O5NONE %s
// O5NONE-NOT: "-O5"
// O5NONE: argument unused during compilation: '-Xarch_i386 -O5'

// RUN: not %clang -target i386-apple-darwin11 -m32 -Xarch_i386 -o -Xarch_i386 -S %s -S -Xarch_i386 -o 2>&1 | FileCheck -check-prefix=INVALID %s
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -o'
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -S'
// INVALID: error: invalid Xarch argument: '-Xarch_i386 -o'
