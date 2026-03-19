// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <algorithm>
#include <array>
#include <numeric>

using namespace sycl;

size_t NumErrors = 0;

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &OS, const std::array<T, N> &Arr) {
  OS << "{";
  for (size_t I = 0; I < N; ++I) {
    if (I)
      OS << ",";
    OS << Arr[I];
  }
  OS << "}";
  return OS;
}

template <typename T, int Dims>
void CheckFill(queue &Q, range<Dims> Range, T Init, T Expected) {
  std::vector<T> Data(Range.size(), Init);
  {
    buffer<T, Dims> Buffer(Data.data(), Range);
    Q.submit([&](handler &CGH) {
       accessor Accessor(Buffer, CGH, write_only);
       CGH.fill(Accessor, Expected);
     }).wait_and_throw();
  }
  for (size_t I = 0; I < Range.size(); ++I) {
    if (Data[I] != Expected) {
      std::cout << "Unexpected value " << Data[I] << " at index " << I
                << " after fill. Expected " << Expected << "." << std::endl;
      ++NumErrors;
      return;
    }
  }
}

template <typename T>
void CheckFillZeroDimAccessor(queue &Q, T Init, T Expected) {
  constexpr int Dims = 1;
  range<1> Range(1);
  std::vector<T> Data(Range.size(), Init);
  {
    buffer<T, Dims> Buffer(Data.data(), Range);
    Q.submit([&](handler &CGH) {
       accessor<T, 0, sycl::access::mode::write> Accessor(Buffer, CGH);
       CGH.fill(Accessor, Expected);
     }).wait_and_throw();
  }
  for (size_t I = 0; I < Range.size(); ++I) {
    if (Data[I] != Expected) {
      std::cout << "Unexpected value " << Data[I] << " at index " << I
                << " after fill. Expected " << Expected << "." << std::endl;
      ++NumErrors;
      return;
    }
  }
}

template <typename T>
void CheckFillDifferentDims(queue &Q, size_t N, T Init, T Expected) {
  CheckFillZeroDimAccessor<T>(Q, Init, Expected);
  CheckFill<T>(Q, range<1>{N}, Init, Expected);
  CheckFill<T>(Q, range<2>{N, N}, Init, Expected);
  CheckFill<T>(Q, range<3>{N, N, N}, Init, Expected);
}

int main() {
  queue Q;

  // Test different power-of-two sizes.
  CheckFillDifferentDims<char>(Q, 10, 'a', 'z');
  CheckFillDifferentDims<std::array<char, 2>>(Q, 10, {'a', 'z'}, {'z', 'a'});
  CheckFillDifferentDims<short>(Q, 10, 8, -16);
  CheckFillDifferentDims<float>(Q, 10, 123.4, 3.14);
  CheckFillDifferentDims<uint64_t>(Q, 10, 42, 24);
  CheckFillDifferentDims<std::array<uint64_t, 2>>(Q, 10, {4, 42}, {24, 4});
  CheckFillDifferentDims<std::array<uint64_t, 4>>(Q, 10, {4, 42, 424, 4242},
                                                  {2424, 424, 24, 4});
  CheckFillDifferentDims<std::array<uint64_t, 8>>(
      Q, 10, {4, 42, 424, 4242, 42424, 424242, 4242424, 42424242},
      {24242424, 2424242, 242424, 24242, 2424, 424, 24, 4});
  CheckFillDifferentDims<std::array<uint64_t, 16>>(
      Q, 10,
      {24242424, 2424242, 242424, 24242, 2424, 424, 24, 4, 4, 42, 424, 4242,
       42424, 424242, 4242424, 42424242},
      {4, 42, 424, 4242, 42424, 424242, 4242424, 42424242, 24242424, 2424242,
       242424, 24242, 2424, 424, 24, 4});

  // Test with non-power-of-two sizes.
  CheckFillDifferentDims<std::array<char, 5>>(Q, 10, {'a', 'b', 'c', 'd', 'e'},
                                              {'A', 'B', 'C', 'D', 'E'});
  std::array<char, 129> InitCharArray129;
  std::fill(InitCharArray129.begin(), InitCharArray129.end(), 130);
  std::array<char, 129> ExpectedCharArray129;
  std::iota(ExpectedCharArray129.begin(), ExpectedCharArray129.end(), 1);
  CheckFillDifferentDims<std::array<char, 129>>(Q, 10, InitCharArray129,
                                                ExpectedCharArray129);

  return NumErrors;
}
