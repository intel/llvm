// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -Xclang -sycl-std=2020 -mllvm -sycl-opt -S -emit-llvm  -o %t_temp.ll %s
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

const size_t N = 10;

template <typename T> class init_a;

template <typename T> bool test(queue myQueue) {
  {
    buffer<T, 1> a(range<1>{N});
    T test = 42;

    myQueue.submit([&](handler &cgh) {
      auto A = a.template get_access<access::mode::write>(cgh);
      cgh.parallel_for<init_a<T>>(range<1>{N},
                                  [=](id<1> index) { A[index] = test; });
    });

    auto A = a.template get_access<access::mode::read>();
    std::cout << "Result:" << std::endl;
    for (size_t i = 0; i < N; i++) {
      if (A[i] != test) {
        std::cout << "ERROR\n";
        return false;
      }
    }
  }

  std::cout << "Good computation!" << std::endl;
  return true;
}

int main() {
  queue q;
  int res1 = test<int>(q);
  int res2 = test<unsigned>(q);
  int res3 = test<float>(q);
  int res4 = test<double>(q);
  if (!(res1 && res2 && res3 && res4)) {
    return 1;
  }
  return 0;
}

// CHECK-DAG: @_ZTS6init_aIiE.NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, i32 {{.*}}%2, ptr {{.*}}%3){{.*}}
// CHECK-DAG: @_ZTS6init_aIjE.NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, i32 {{.*}}%2, ptr {{.*}}%3){{.*}}
// CHECK-DAG: @_ZTS6init_aIfE.NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, float {{.*}}%2, ptr {{.*}}%3){{.*}}
// CHECK-DAG: @_ZTS6init_aIdE.NativeCPUKernel(ptr {{.*}}%0, ptr {{.*}}%1, double {{.*}}%2, ptr {{.*}}%3){{.*}}
