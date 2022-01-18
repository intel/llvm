// RUN: %clangxx -S -fsycl -fsycl-device-only %s
#include <CL/sycl.hpp>

int main() {
  long long int a = -9223372034707292160ll;
  auto b = sycl::ctz(a);
  return 0;
  // CHECK: call spir_func i32 {{.*}}__sycl_std::__invoke_ctz{{.*}} i32 addrspace(1)*
}
