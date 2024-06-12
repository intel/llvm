// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

#include <iostream>

struct user_defined_type {
  float a;
  // this field causes a padding to be inserted in the middle of a struct
  char d;
  int b;
  char c;

  constexpr user_defined_type(float a, int b, char c)
      : a(a), d('a'), b(b), c(c) {}
  constexpr user_defined_type(const user_defined_type &) = default;

  bool operator==(const user_defined_type &other) const {
    return other.a == a && other.b == b && other.c == c && other.d == d;
  }

  void dump() const {
    std::cout << "user_defined_type {" << std::endl;
    std::cout << "\ta = " << a << "\n\td = " << d << std::endl;
    std::cout << "\tb = " << b << "\n\tc = " << c << std::endl;
    std::cout << "}" << std::endl;
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

  constexpr user_defined_type new_data(1.0, 2, 3);
  {
    sycl::buffer buf(&data, sycl::range<1>{1});
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access(cgh);
       cgh.set_specialization_constant<spec_id>(new_data);
       cgh.single_task([=](sycl::kernel_handler kh) {
         acc[0] = kh.get_specialization_constant<spec_id>();
       });
     }).wait();
  }

  new_data.dump();
  data.dump();
  assert(new_data == data);

  return 0;
}
