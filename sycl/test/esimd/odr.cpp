// The test checks that ESIMD headers don't break the ODR:
// two SYCL sources including ESIMD headers can be compiled and linked into a
// single executable w/o linker complaining about multiple symbol definitions.
//
// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-targets=%sycl_triple -DSOURCE1 -c %s -o %t1.o
// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-targets=%sycl_triple -DSOURCE2 -c %s -o %t2.o
// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-targets=%sycl_triple %t1.o %t2.o -o %t.exe
//

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

#ifdef SOURCE1
void run_kernel2(queue &, int *);

int main() {
  queue q;
  int *data = static_cast<int *>(
      malloc_shared(sizeof(int), q.get_device(), q.get_context()));
  *data = 5;

  try {
    q.submit([&](handler &cgh) {
      cgh.single_task<class my_kernel>([=]() SYCL_ESIMD_KERNEL { data[0]++; });
    });
    q.wait();
    run_kernel2(q, data);
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 2;
  }
  if (*data == 7) {
    std::cout << "Passed\n";
    return 0;
  } else {
    std::cout << "Failed: " << *data << "!= 7(gold)\n";
    return 1;
  }
}
#elif defined SOURCE2
void run_kernel2(queue &q, int *data) {
  try {
    q.submit([&](handler &cgh) {
      cgh.single_task<class my_kernel>([=]() SYCL_ESIMD_KERNEL { data[0]++; });
    });
    q.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
  }
}
#endif
