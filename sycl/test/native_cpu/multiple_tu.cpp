// REQUIRES: native_cpu
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu %s -c -o %t_main.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu %S/Inputs/init.cpp -c -o %t_init.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu %S/Inputs/plusone.cpp -c -o %t_plusone.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t_plusone.o %t_main.o %t_init.o -o %t
//RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t

// Re-run test with -g
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu -g %s -c -o %t_main-debug.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu -g %S/Inputs/init.cpp -c -o %t_init-debug.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu -g %S/Inputs/plusone.cpp -c -o %t_plusone-debug.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu -g %t_plusone-debug.o %t_main-debug.o %t_init-debug.o -o %t-debug
//RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t-debug

// New offload driver
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu --offload-new-driver %s -c -o %t_main-new_offload.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu --offload-new-driver %S/Inputs/init.cpp -c -o %t_init-new_offload.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu --offload-new-driver %S/Inputs/plusone.cpp -c -o %t_plusone-new_offload.o
//RUN: %clangxx -fsycl -fsycl-targets=native_cpu --offload-new-driver %t_plusone-new_offload.o %t_main-new_offload.o %t_init-new_offload.o -o %t-new_offload
//RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t-new_offload

#include "Inputs/common.h"
#include <iostream>

int main() {
  const size_t size = 10;
  sycl::queue q;
  int *data = sycl::malloc_device<int>(size, q);
  init(data, size, q);
  plusone(data, size, q);
  int res[size];
  q.memcpy(res, data, size * sizeof(int)).wait();
  for (auto &el : res) {
    if (el != 42) {
      std::cout << "Error, " << el << " is not 42\n";
    }
  }
  std::cout << std::endl;
  sycl::free(data, q);
}
