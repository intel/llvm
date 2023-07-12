#include <sycl/sycl.hpp>
#include <vector>

void init(int *data, size_t size, sycl::queue q);
void plusone(int *data, size_t size, sycl::queue q);
