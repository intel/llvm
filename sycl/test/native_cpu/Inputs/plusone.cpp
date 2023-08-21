#include "common.h"

void plusone(int *data, size_t size, sycl::queue q) {
  q.parallel_for(size, [=](sycl::id<1> id) { data[id] = data[id] + 1; }).wait();
}
