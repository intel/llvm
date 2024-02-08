#include "sycl/sycl.hpp"

void add_x_1(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_x_1_dummy>([=] { *result = a + b + 1; });
}

void add_x_2(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_x_2_dummy>([=] { *result = a + b + 2; });
}

void add_x_3(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_x_3_dummy>([=] { *result = a + b + 3; });
}

void add_x_4(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_x_4_dummy>([=] { *result = a + b + 4; });
}

void add_x_5(sycl::queue q, int *result, int a, int b) {
  q.single_task<class add_x_5_dummy>([=] { *result = a + b + 5; });
}
