// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

template <int Dimensions> sycl::range<Dimensions> create_range() {
  return sycl::range<Dimensions>(1);
}

template <> sycl::range<2> create_range() { return sycl::range<2>(1, 1); }

template <> sycl::range<3> create_range() { return sycl::range<3>(1, 1, 1); }

// Compile only test to check that buffer_allocator type does not get
// reinterpreted with const keyworkd
template <typename T, int Dimensions,
          typename Allocator = sycl::buffer_allocator<T>>
void test_buffer_const_reinterpret() {
  sycl::buffer<T, Dimensions, Allocator> buff(create_range<Dimensions>());
  sycl::buffer<T const, Dimensions, Allocator> const_buff(
      create_range<Dimensions>());

  auto reinterpret_buff = buff.template reinterpret<T const, Dimensions>(
      create_range<Dimensions>());

  static_assert(
      std::is_same_v<decltype(const_buff), decltype(reinterpret_buff)>);
}

struct my_struct {
  int my_int = 0;
  float my_float = 0;
  double my_double = 0;
};

int main() {
  test_buffer_const_reinterpret<short, 1>();
  test_buffer_const_reinterpret<int, 1>();
  test_buffer_const_reinterpret<long, 1>();
  test_buffer_const_reinterpret<unsigned short, 1>();
  test_buffer_const_reinterpret<unsigned int, 1>();
  test_buffer_const_reinterpret<unsigned long, 1>();
  test_buffer_const_reinterpret<sycl::half, 1>();
  test_buffer_const_reinterpret<float, 1>();
  test_buffer_const_reinterpret<double, 1>();
  test_buffer_const_reinterpret<my_struct, 1>();

  test_buffer_const_reinterpret<short, 2>();
  test_buffer_const_reinterpret<int, 2>();
  test_buffer_const_reinterpret<long, 2>();
  test_buffer_const_reinterpret<unsigned short, 2>();
  test_buffer_const_reinterpret<unsigned int, 2>();
  test_buffer_const_reinterpret<unsigned long, 2>();
  test_buffer_const_reinterpret<sycl::half, 2>();
  test_buffer_const_reinterpret<float, 2>();
  test_buffer_const_reinterpret<double, 2>();
  test_buffer_const_reinterpret<my_struct, 2>();

  test_buffer_const_reinterpret<short, 3>();
  test_buffer_const_reinterpret<int, 3>();
  test_buffer_const_reinterpret<long, 3>();
  test_buffer_const_reinterpret<unsigned short, 3>();
  test_buffer_const_reinterpret<unsigned int, 3>();
  test_buffer_const_reinterpret<unsigned long, 3>();
  test_buffer_const_reinterpret<sycl::half, 3>();
  test_buffer_const_reinterpret<float, 3>();
  test_buffer_const_reinterpret<double, 3>();
  test_buffer_const_reinterpret<my_struct, 3>();

  return 0;
}
