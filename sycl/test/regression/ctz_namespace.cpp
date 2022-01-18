// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  long long int a = -9223372034707292160ll;
  auto b = sycl::ctz(a);
  return 0;
  // CHECK: call spir_func i32 {{.*}}__sycl_std::__invoke_ctz{{.*}} i32 addrspace(1)*
}
