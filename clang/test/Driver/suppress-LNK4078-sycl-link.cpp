// RUN:  %clangxx -target x86_64-pc-windows-msvc -fsycl %s -### 2>&1 \
// RUN:  | FileCheck %s
// RUN:  %clang_cl --target=x86_64-pc-windows-msvc -fsycl %s -### 2>&1 \
// RUN:  | FileCheck %s
// CHECK: /IGNORE:4078
