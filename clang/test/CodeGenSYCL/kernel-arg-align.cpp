// RUN: %clang_cc1 -fsycl-is-device -O0 -internal-isystem %S/Inputs -triple spir64 -emit-llvm -o - %s | FileCheck %s

// Test that the pointer parameters generated for the kernel do not
// have alignment on them.

#include "sycl.hpp"

using namespace sycl;

struct S;

void Test() {
  struct MyIP {
    char* a;
    int* b;
    double* c;

    void operator()() const {
       *((int *) a)  = 1; // 1 on arg, 4 on site
       *((double *) b)  = 2; // 4 on arg, 8 on site
       *((char *) c)  = 3; // 8 on arg, 1 on site
    }
  };

  constexpr int kN = 8;
  auto host_array_A =
      malloc_shared<char>(kN);

  auto host_array_B =
      malloc_shared<int>(kN);

  auto host_array_C =
      malloc_shared<double>(kN);

  for (int i = 0; i < kN; i++) {
    host_array_A[i] = i;
    host_array_B[i] = i * 2;
  }

  sycl::kernel_single_task<S>(MyIP{host_array_A, host_array_B, host_array_C});

  free(host_array_A);
  free(host_array_B);
  free(host_array_C);
}

int main() {
  Test();
  return 0;
}

// CHECK: define {{.*}} spir_kernel void @_ZTS1S(ptr addrspace(1) noundef %_arg_a, ptr addrspace(1) noundef %_arg_b, ptr addrspace(1) noundef %_arg_c)
