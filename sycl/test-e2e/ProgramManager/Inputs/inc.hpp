#include <iostream>
#include <sycl/detail/kernel_properties.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/kernel_execution_properties.hpp>
#include <sycl/sycl.hpp>

class KernelTest1;
class KernelTest2;

void gpu_foo(sycl::queue &queue, int *buf);
void gpu_bar(sycl::queue &queue, int *buf);
