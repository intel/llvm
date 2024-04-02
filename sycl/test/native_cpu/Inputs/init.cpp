#include "common.h"
#include <vector>

void init(int *data, size_t size, sycl::queue q) {
  q.parallel_for(size, [=](sycl::id<1> id) { data[id] = 41; }).wait();
}
