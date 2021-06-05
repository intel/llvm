// REQUIRES: system-windows
// RUN:  %clangxx -target x86_64-pc-windows-msvc -fsycl -fintelfpga -fsycl-link -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s
// RUN:  %clang_cl -target x86_64-pc-windows-msvc -fsycl -fintelfpga -fsycl-link -Xshardware %s -### 2>&1 \
// RUN:  | FileCheck %s
// CHECK: /IGNORE:4221
