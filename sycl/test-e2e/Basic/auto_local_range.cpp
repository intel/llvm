// RUN: %{build} -I . -o %t.out
// RUN: %{run} %t.out

#include <cstdlib>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/auto_local_range.hpp>

template <size_t... Args> bool testAutoLocalRange() {
  constexpr int Dimensions = sizeof...(Args);
  static_assert(1 <= Dimensions && Dimensions <= 3);
  using namespace sycl;

  queue q;
  buffer<size_t, Dimensions> data{range<Dimensions>{Args...}};
  q.submit([&](handler &cgh) {
     const accessor acc{data, cgh, write_only};
     cgh.parallel_for(
         nd_range<Dimensions>{
             range<Dimensions>{Args...},
             sycl::ext::oneapi::experimental::auto_range<Dimensions>()},
         [=](auto id) { acc[id.get_global_id()] = id.get_global_linear_id(); });
   }).wait();

  size_t count = 0;
  const host_accessor<size_t, Dimensions, access_mode::read> acc{data};
  for (const auto &value : acc) {
    if (value != count) {
      return false;
    }
    count++;
  }
  return count == acc.size();
}

int main() {
  static_assert(SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE == 1,
                "SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE must have a value of 1");

#define TEST_CASE(NAME, ARGS...)                                               \
  {                                                                            \
    const bool passed = testAutoLocalRange<ARGS>();                            \
    if (!passed) {                                                             \
      std::cerr << "Test " << NAME << " failed" << std::endl;                  \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

  TEST_CASE("A0", 1)
  TEST_CASE("A1", 10)
  TEST_CASE("A2", 256)

  TEST_CASE("B0", 1, 1)
  TEST_CASE("B1", 10, 1)
  TEST_CASE("B2", 10, 10)
  TEST_CASE("B3", 32, 10)
  TEST_CASE("B4", 32, 32)

  TEST_CASE("C0", 1, 1, 1)
  TEST_CASE("C1", 10, 1, 1)
  TEST_CASE("C2", 10, 10, 10)
  TEST_CASE("C3", 32, 10, 10)
  TEST_CASE("C4", 2, 32, 32)

#undef TEST_CASE
  std::cout << "Tests passed!" << std::endl;
  return EXIT_SUCCESS;
}
