// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

struct user_defined_type {
  float a;
  int b;
  char c;

  constexpr user_defined_type(float a, int b, char c) : a(a), b(b), c(c) {}
  constexpr user_defined_type(const user_defined_type &) = default;

  bool operator==(const user_defined_type &other) const {
    return other.a == a && other.b == b && other.c == c;
  }
};

constexpr user_defined_type reference(3.14, 42, 8);
constexpr sycl::specialization_id<user_defined_type> spec_id(reference);

int main() {
  sycl::queue q;
  user_defined_type data(0, 0, 0);

  {
    sycl::buffer buf(&data, sycl::range<1>{1});
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access(cgh);
       cgh.single_task([=](sycl::kernel_handler kh) {
         acc[0] = kh.get_specialization_constant<spec_id>();
       });
     }).wait();
  }

  assert(reference == data);

  return 0;
}
