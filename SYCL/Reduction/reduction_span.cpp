// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// `Group algorithms are not supported on host device.` on HIP backend.
// XFAIL: hip

// This test performs basic checks of reductions initialized with a sycl::span

#include <sycl/sycl.hpp>
using namespace sycl;

int NumErrors = 0;

template <int Dimensions> size_t getLinearSize(range<Dimensions> Range) {
  return Range.size();
}

template <int Dimensions> size_t getLinearSize(nd_range<Dimensions> NDRange) {
  return NDRange.get_global_range().size();
}

template <int Dimensions>
size_t getLinearId(nd_range<Dimensions>, nd_item<Dimensions> Item) {
  return Item.get_global_linear_id();
}

size_t getLinearId(range<1>, id<1> Id) { return Id[0]; }

size_t getLinearId(range<2> Range, id<2> Id) {
  return Id[0] * Range[1] + Id[1];
}

size_t getLinearId(range<3> Range, id<3> Id) {
  return Id[0] * Range[1] * Range[2] + Id[1] * Range[2] + Id[2];
}

enum class submission_mode {
  handler,
  queue,
};

template <size_t N, typename T, typename BinaryOperation, typename Range,
          submission_mode SubmissionMode>
void test(queue Q, Range Rng, T Identity, T Value) {

  // Initialize output to identity value
  T *Output = malloc_shared<T>(N, Q);
  Q.parallel_for(range<1>{N}, [=](id<1> I) { Output[I] = Identity; }).wait();

  // Perform generalized "histogram" with N bins
  auto Redu = reduction(span<T, N>(Output, N), Identity, BinaryOperation());
  auto Kern = [=](auto It, auto &Reducer) {
    size_t Index = getLinearId(Rng, It) % N;
    Reducer[Index].combine(Value);
  };
  if constexpr (SubmissionMode == submission_mode::handler) {
    Q.submit([&](handler &CGH) { CGH.parallel_for(Rng, Redu, Kern); }).wait();
  } else /*if (SubmissionMode == submission_mode::queue) */ {
    Q.parallel_for(Rng, Redu, Kern).wait();
  }

  size_t Size = getLinearSize(Rng);

  // Each bin should have the same value unless B doesn't divide N
  T Expected = Identity;
  T ExpectedRemainder;
  for (size_t I = 0; I < Size; I += N) {
    ExpectedRemainder = Expected;
    Expected = BinaryOperation()(Expected, Value);
  }

  bool Passed = true;
  for (size_t I = 0; I < N; ++I) {
    if (I < Size % N) {
      Passed &= (Output[I] == Expected);
    } else {
      Passed &= (Output[I] == ExpectedRemainder);
    }
  }

  free(Output, Q);
  NumErrors += (Passed) ? 0 : 1;
}

struct CustomType {
  int x;
  bool operator==(const CustomType &o) const { return (x == o.x); }
};

struct CustomBinaryOperation {
  CustomType operator()(const CustomType &lhs, const CustomType &rhs) const {
    return CustomType{lhs.x + rhs.x};
  }
};

int main() {
  queue Q;

  // Tests for small spans that can be privatized efficiently
  // Each combination tests a different sycl::reduction implementation
  test<16, int, std::plus<int>, sycl::range<1>, submission_mode::handler>(Q, 24,
                                                                          0, 1);
  test<16, float, std::plus<float>, sycl::range<1>, submission_mode::handler>(
      Q, 24, 0, 1);
  test<16, int, std::multiplies<int>, sycl::range<1>, submission_mode::handler>(
      Q, 24, 1, 2);
  test<16, CustomType, CustomBinaryOperation, sycl::range<1>,
       submission_mode::handler>(Q, 24, CustomType{0}, CustomType{1});
  test<16, int, std::plus<int>, sycl::range<1>, submission_mode::queue>(Q, 24,
                                                                        0, 1);
  test<16, float, std::plus<float>, sycl::range<1>, submission_mode::queue>(
      Q, 24, 0, 1);
  test<16, int, std::multiplies<int>, sycl::range<1>, submission_mode::queue>(
      Q, 24, 1, 2);
  test<16, CustomType, CustomBinaryOperation, sycl::range<1>,
       submission_mode::queue>(Q, 24, CustomType{0}, CustomType{1});

  test<16, int, std::plus<int>, sycl::nd_range<1>, submission_mode::handler>(
      Q, {24, 8}, 0, 1);
  test<16, float, std::plus<float>, sycl::nd_range<1>,
       submission_mode::handler>(Q, {24, 8}, 0, 1);
  test<16, int, std::multiplies<int>, sycl::nd_range<1>,
       submission_mode::handler>(Q, {24, 8}, 1, 2);
  test<16, int, std::bit_or<int>, sycl::nd_range<1>, submission_mode::handler>(
      Q, {24, 8}, 0, 1);
  test<16, CustomType, CustomBinaryOperation, sycl::nd_range<1>,
       submission_mode::handler>(Q, {24, 8}, CustomType{0}, CustomType{1});
  test<16, int, std::plus<int>, sycl::nd_range<1>, submission_mode::queue>(
      Q, {24, 8}, 0, 1);
  test<16, float, std::plus<float>, sycl::nd_range<1>, submission_mode::queue>(
      Q, {24, 8}, 0, 1);
  test<16, int, std::multiplies<int>, sycl::nd_range<1>,
       submission_mode::queue>(Q, {24, 8}, 1, 2);
  test<16, int, std::bit_or<int>, sycl::nd_range<1>, submission_mode::queue>(
      Q, {24, 8}, 0, 1);
  test<16, CustomType, CustomBinaryOperation, sycl::nd_range<1>,
       submission_mode::queue>(Q, {24, 8}, CustomType{0}, CustomType{1});

  return NumErrors;
}
