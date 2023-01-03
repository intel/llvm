//

// RUN: %clang_cc1 -emit-llvm -mdebug-pass=Structure -o /dev/null 2>&1 %s | FileCheck %s --check-prefix=

//