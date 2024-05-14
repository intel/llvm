// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
#include <sycl/detail/core.hpp>

#include <sycl/specialization_id.hpp>

#include <iostream>

struct nested {
  char a, b, c;
  short d;

  constexpr nested(char a) : a(a), b(a), c(a), d(a) {}
  constexpr nested(const nested &) = default;

  bool operator==(const nested &other) const {
    return other.a == a && other.b == b && other.c == c && other.d == d;
  }
};

struct alignas(32) user_defined_type2 {
  float a;
  char b;
  int c;

  constexpr user_defined_type2(float a, char b, int c) : a(a), b(b), c(c) {}
  constexpr user_defined_type2(const user_defined_type2 &) = default;

  bool operator==(const user_defined_type2 &other) const {
    return other.a == a && other.b == b && other.c == c;
  }
};

struct user_defined_type {
  float a;
  user_defined_type2 n;
  int b alignas(32);
  char c;

  constexpr user_defined_type(float a, int b, char c)
      : a(a), n(a, c, b), b(b), c(c) {}
  constexpr user_defined_type(const user_defined_type &) = default;

  bool operator==(const user_defined_type &other) const {
    return other.a == a && other.b == b && other.c == c && other.n == n;
  }
};

struct user_defined_type3 {
  char x = 'x';
  struct {
    char y = 'y';
    int z = 'z';
    char a = 'a';
  } s;
  char b = 'b';
  bool operator==(const user_defined_type3 &rhs) const {
    return x == rhs.x && s.y == rhs.s.y && s.z == rhs.s.z && s.a == rhs.s.a &&
           b == rhs.b;
  }
};

constexpr user_defined_type reference(3.14, 42, 8);
constexpr sycl::specialization_id<user_defined_type> spec_id(reference);

constexpr user_defined_type2 reference2(3.14, 42, 8);
constexpr sycl::specialization_id<user_defined_type2> spec_id2(reference2);

constexpr user_defined_type3 reference3{};
constexpr sycl::specialization_id<user_defined_type3> spec_id3(reference3);

int main() {
  sycl::queue q;

  user_defined_type2 data2(0, 0, 0);

  {
    sycl::buffer buf(&data2, sycl::range<1>{1});
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access(cgh);
       cgh.single_task([=](sycl::kernel_handler kh) {
         acc[0] = kh.get_specialization_constant<spec_id2>();
       });
     }).wait();
  }

  assert(reference2 == data2);

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

  assert(new_data == data);

  user_defined_type3 data3;
  {
    sycl::buffer buf(&data3, sycl::range<1>{1});
    q.submit([&](sycl::handler &cgh) {
       auto acc = buf.get_access(cgh);
       cgh.single_task([=](sycl::kernel_handler kh) {
         acc[0] = kh.get_specialization_constant<spec_id3>();
       });
     }).wait();
  }
  assert(reference3 == data3);

  return 0;
}
