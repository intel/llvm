// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out

// This performs basic checks such as reduction creation, getIdentity() method,
// and the combine() method of the aux class 'reducer'.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;


template <typename T, typename Reduction>
void test_reducer(Reduction &Redu, T A, T B) {
  typename Reduction::reducer_type Reducer;
  Reducer.combine(A);
  Reducer.combine(B);

  typename Reduction::binary_operation BOp;
  T ExpectedValue = BOp(A, B);
  assert(ExpectedValue == Reducer.MValue &&
         "Wrong result of binary operation.");
}

template <typename T, typename Reduction, typename BinaryOperation>
void test_reducer(Reduction &Redu, T Identity, BinaryOperation BOp, T A, T B) {
  typename Reduction::reducer_type Reducer(Identity, BOp);
  Reducer.combine(A);
  Reducer.combine(B);

  T ExpectedValue = BOp(A, B);
  assert(ExpectedValue == Reducer.MValue &&
         "Wrong result of binary operation.");
}

template <typename T, int Dim, class BinaryOperation>
class Known;
template <typename T, int Dim, class BinaryOperation>
class Unknown;

template <typename T, int Dim, class BinaryOperation>
void testKnown(T Identity, BinaryOperation BOp, T A, T B) {
  buffer<T, 1> ReduBuf(1);

  queue Q;
  Q.submit([&](handler &CGH) {
    // Reduction needs a global_buffer accessor as a parameter.
    // This accessor is not really used in this test.
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        ReduAcc(ReduBuf, CGH);
    auto Redu = intel::reduction(ReduAcc, BOp);
    assert(Redu.getIdentity() == Identity &&
           "Failed getIdentity() check().");
    test_reducer(Redu, A, B);
    test_reducer(Redu, Identity, BOp, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<Known<T, Dim, BinaryOperation>>([=]() {});
  });
}

template <typename T, int Dim, typename KernelName, class BinaryOperation>
void testUnknown(T Identity, BinaryOperation BOp, T A, T B) {
  buffer<T, 1> ReduBuf(1);
  queue Q;
  Q.submit([&](handler &CGH) {
    // Reduction needs a global_buffer accessor as a parameter.
    // This accessor is not really used in this test.
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        ReduAcc(ReduBuf, CGH);
    auto Redu = intel::reduction(ReduAcc, Identity, BOp);
    assert(Redu.getIdentity() == Identity &&
           "Failed getIdentity() check().");
    test_reducer(Redu, Identity, BOp, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<KernelName>([=]() {});
  });
}

template <typename T, class BinaryOperation>
void testBoth(T Identity, BinaryOperation BOp, T A, T B) {
  testKnown<T, 0>(Identity, BOp, A, B);
  testKnown<T, 1>(Identity, BOp, A, B);
  testUnknown<T, 0, Unknown<T, 0, BinaryOperation>>(Identity, BOp, A, B);
  testUnknown<T, 1, Unknown<T, 1, BinaryOperation>>(Identity, BOp, A, B);
}

int main() {
  testBoth<int>(0, intel::plus<int>(), 1, 7);
  testBoth<int>(1, std::multiplies<int>(), 1, 7);
  testBoth<int>(0, intel::bit_or<int>(), 1, 8);
  testBoth<int>(0, intel::bit_xor<int>(), 7, 3);
  testBoth<int>(~0, intel::bit_and<int>(), 7, 3);
  testBoth<int>((std::numeric_limits<int>::max)(), intel::minimum<int>(), 7, 3);
  testBoth<int>((std::numeric_limits<int>::min)(), intel::maximum<int>(), 7, 3);

  testBoth<float>(0, intel::plus<float>(), 1, 7);
  testBoth<float>(1, std::multiplies<float>(), 1, 7);
  testBoth<float>(getMaximumFPValue<float>(), intel::minimum<float>(), 7, 3);
  testBoth<float>(getMinimumFPValue<float>(), intel::maximum<float>(), 7, 3);

  testUnknown<CustomVec<float>, 0,
              Unknown<CustomVec<float>, 0, CustomVecPlus<float>>>(
      CustomVec<float>(0), CustomVecPlus<float>(), CustomVec<float>(1),
      CustomVec<float>(7));
  testUnknown<CustomVec<float>, 1,
              Unknown<CustomVec<float>, 1, CustomVecPlus<float>>>(
      CustomVec<float>(0), CustomVecPlus<float>(), CustomVec<float>(1),
      CustomVec<float>(7));

  testUnknown<int, 0, class BitOrName>(
      0, [](auto a, auto b) { return a | b; }, 1, 8);

  std::cout << "Test passed\n";
  return 0;
}
