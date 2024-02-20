#include "sycl/sycl.hpp"

void sub(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_dummy>([=] { *result = a - b; });
}
