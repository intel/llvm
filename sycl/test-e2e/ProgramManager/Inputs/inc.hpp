#include <sycl/detail/core.hpp>

class KernelTest1;
class KernelTest2;

void gpu_foo(sycl::queue &queue, int *buf);
void gpu_bar(sycl::queue &queue, int *buf);
