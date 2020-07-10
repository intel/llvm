// REQUIRES: system-windows
// RUN:  %clangxx -fsycl -fintelfpga -fsycl-link -Xshardware %s 2>&1 \
// RUN:  | FileCheck %s
// CHECK: No kernels specified.
// CHECK-NOT: LNK4221
int main() {
  return 0;
}
