// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// `Group algorithms are not supported on host device.` on Nvidia.
// XFAIL: hip_nvidia

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows
// REQUIRES: aspect-usm_shared_allocations
// This test performs basic checks of reductions initialized with a pack
// containing at least one sycl::span

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

// Test a span and a regular sum
template <size_t N, typename T, typename BinaryOperation, typename Range,
          submission_mode SubmissionMode>
void test1(queue Q, Range Rng, T Identity, T Value) {

  // Initialize output to identity value
  int *Sum = malloc_shared<int>(1, Q);
  Q.single_task([=]() { *Sum = 0; }).wait();
  T *Output = malloc_shared<T>(N, Q);
  Q.parallel_for(range<1>{N}, [=](id<1> I) { Output[I] = Identity; }).wait();

  // Perform generalized "histogram" with N bins
  auto ScalarRedu = reduction(Sum, plus<>());
  auto SpanRedu = reduction(span<T, N>(Output, N), Identity, BinaryOperation());
  auto Kern = [=](auto It, auto &ScalarReducer, auto &SpanReducer) {
    ScalarReducer++;
    size_t Index = getLinearId(Rng, It) % N;
    SpanReducer[Index].combine(Value);
  };
  if constexpr (SubmissionMode == submission_mode::handler) {
    Q.submit([&](handler &CGH) {
       CGH.parallel_for(Rng, ScalarRedu, SpanRedu, Kern);
     }).wait();
  } else /*if (SubmissionMode == submission_mode::queue) */ {
    Q.parallel_for(Rng, ScalarRedu, SpanRedu, Kern).wait();
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
  Passed &= (*Sum == Size);

  free(Output, Q);
  free(Sum, Q);
  NumErrors += (Passed) ? 0 : 1;
}

// Test two spans
template <size_t N, typename T, typename BinaryOperation, typename Range,
          submission_mode SubmissionMode>
void test2(queue Q, Range Rng, T Identity, T Value) {

  // Initialize output to identity value
  int *Output1 = malloc_shared<int>(N, Q);
  Q.parallel_for(range<1>{N}, [=](id<1> I) { Output1[I] = 0; }).wait();
  T *Output2 = malloc_shared<T>(N, Q);
  Q.parallel_for(range<1>{N}, [=](id<1> I) { Output2[I] = Identity; }).wait();

  // Perform generalized "histogram" with N bins
  auto Redu1 = reduction(span<int, N>(Output1, N), plus<>());
  auto Redu2 = reduction(span<T, N>(Output2, N), Identity, BinaryOperation());
  auto Kern = [=](auto It, auto &Reducer1, auto &Reducer2) {
    size_t Index = getLinearId(Rng, It) % N;
    Reducer1[Index]++;
    Reducer2[Index].combine(Value);
  };
  if constexpr (SubmissionMode == submission_mode::handler) {
    Q.submit([&](handler &CGH) {
       CGH.parallel_for(Rng, Redu1, Redu2, Kern);
     }).wait();
  } else /*if (SubmissionMode == submission_mode::queue) */ {
    Q.parallel_for(Rng, Redu1, Redu2, Kern).wait();
  }

  size_t Size = getLinearSize(Rng);
  bool Passed = true;
  // Span1
  {
    int Expected = 0;
    int ExpectedRemainder;
    for (size_t I = 0; I < Size; I += N) {
      ExpectedRemainder = Expected;
      Expected += 1;
    }

    for (size_t I = 0; I < N; ++I) {
      if (I < Size % N) {
        Passed &= (Output1[I] == Expected);
      } else {
        Passed &= (Output1[I] == ExpectedRemainder);
      }
    }
  }

  // Span2
  {
    T Expected = Identity;
    T ExpectedRemainder;
    for (size_t I = 0; I < Size; I += N) {
      ExpectedRemainder = Expected;
      Expected = BinaryOperation()(Expected, Value);
    }

    for (size_t I = 0; I < N; ++I) {
      if (I < Size % N) {
        Passed &= (Output2[I] == Expected);
      } else {
        Passed &= (Output2[I] == ExpectedRemainder);
      }
    }
  }

  free(Output2, Q);
  free(Output1, Q);
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

template <size_t N, typename T, typename BinaryOperation, typename Range,
          submission_mode SubmissionMode>
void test(queue Q, Range Rng, T Identity, T Value) {
  test1<N, T, BinaryOperation, Range, SubmissionMode>(Q, Rng, Identity, Value);
  test2<N, T, BinaryOperation, Range, SubmissionMode>(Q, Rng, Identity, Value);
}

int main() {
  queue Q;

  // Tests for small spans that can be privatized efficiently
  // Each combination tests a different sycl::reduction implementation
  // TODO: Enable range<> tests once parallel_for accepts pack
  /*test<16, int, std::plus<int>, sycl::range<1>, submission_mode::handler>(Q,
  24, 0, 1); test<16, float, std::plus<float>, sycl::range<1>,
  submission_mode::handler>(Q, 24, 0, 1); test<16, int, std::multiplies<int>,
  sycl::range<1>, submission_mode::handler>(Q, 24, 1, 2); test<16, CustomType,
  CustomBinaryOperation, sycl::range<1>, submission_mode::handler>(Q, 24,
  CustomType{0}, CustomType{1});
  test<16, int, std::plus<int>, sycl::range<1>, submission_mode::queue>(Q, 24,
  0, 1); test<16, float, std::plus<float>, sycl::range<1>,
  submission_mode::queue>(Q, 24, 0, 1); test<16, int, std::multiplies<int>,
  sycl::range<1>, submission_mode::queue>(Q, 24, 1, 2); test<16, CustomType,
  CustomBinaryOperation, sycl::range<1>, submission_mode::queue>(Q, 24,
  CustomType{0}, CustomType{1});*/

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
