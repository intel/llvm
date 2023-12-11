#include "sycl/sycl.hpp"

void add(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_dummy>([=] { *result = a + b; });
}
