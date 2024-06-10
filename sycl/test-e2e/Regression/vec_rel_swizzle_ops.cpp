// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <cstdlib>
#include <sycl/detail/core.hpp>
#include <sycl/types.hpp>

template <typename T, typename ResultT>
bool testAndOperator(const std::string &typeName) {
  constexpr int N = 5;
  std::array<ResultT, N> results{};

  sycl::queue q;
  sycl::buffer<ResultT, 1> buffer{results.data(), N};
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor acc{buffer, cgh, sycl::write_only};
     cgh.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> id) {
       auto testVec1 = sycl::vec<T, 1>(static_cast<T>(1));
       auto testVec2 = sycl::vec<T, 1>(static_cast<T>(2));
       sycl::vec<ResultT, 1> resVec;

       ResultT expected = static_cast<ResultT>(
           -(static_cast<ResultT>(1) && static_cast<ResultT>(2)));
       acc[0] = expected;

       // LHS swizzle
       resVec = testVec1.template swizzle<sycl::elem::s0>() && testVec2;
       acc[1] = resVec[0];

       // RHS swizzle
       resVec = testVec1 && testVec2.template swizzle<sycl::elem::s0>();
       acc[2] = resVec[0];

       // No swizzle
       resVec = testVec1 && testVec2;
       acc[3] = resVec[0];

       // Both swizzle
       resVec = testVec1.template swizzle<sycl::elem::s0>() &&
                testVec2.template swizzle<sycl::elem::s0>();
       acc[4] = resVec[0];
     });
   }).wait();

  bool passed = true;
  ResultT expected = results[0];

  std::cout << "Testing with T = " << typeName << std::endl;
  std::cout << "Expected: " << (int)expected << std::endl;
  for (int i = 1; i < N; i++) {
    std::cout << "Test " << (i - 1) << ": " << ((int)results[i]) << std::endl;
    passed &= expected == results[i];
  }
  std::cout << std::endl;
  return passed;
}

int main() {
  bool passed = true;
  passed &= testAndOperator<bool, std::int8_t>("bool");
  passed &= testAndOperator<std::int8_t, std::int8_t>("std::int8_t");
  passed &= testAndOperator<float, std::int32_t>("float");
  passed &= testAndOperator<int, std::int32_t>("int");
  std::cout << (passed ? "Pass" : "Fail") << std::endl;
  return (passed ? EXIT_SUCCESS : EXIT_FAILURE);
}
