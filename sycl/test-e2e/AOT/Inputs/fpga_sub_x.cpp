#include <sycl/detail/core.hpp>

void sub_x_1(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_x_1_dummy>([=] { *result = a - b - 1; });
}

void sub_x_2(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_x_2_dummy>([=] { *result = a - b - 2; });
}

void sub_x_3(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_x_3_dummy>([=] { *result = a - b - 3; });
}

void sub_x_4(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_x_4_dummy>([=] { *result = a - b - 4; });
}

void sub_x_5(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_x_5_dummy>([=] { *result = a - b - 5; });
}

void sub_x_6(sycl::queue q, int *result, int a, int b) {
  q.single_task<class sub_x_6_dummy>([=] { *result = a - b - 6; });
}
