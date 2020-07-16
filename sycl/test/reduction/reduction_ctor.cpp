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

template <typename T, typename Reduction>
void test_reducer(Reduction &Redu, T Identity, T A, T B) {
  typename Reduction::reducer_type Reducer(Identity);
  Reducer.combine(A);
  Reducer.combine(B);

  typename Reduction::binary_operation BOp;
  T ExpectedValue = BOp(A, B);
  assert(ExpectedValue == Reducer.MValue &&
         "Wrong result of binary operation.");
}

template <typename... Ts>
class KernelNameGroup;

template <typename T>
struct Point {
  Point() : X(0), Y(0) {}
  Point(T X, T Y) : X(X), Y(Y) {}
  Point(T V) : X(V), Y(V) {}
  bool operator==(const Point &P) const {
    return P.X == X && P.Y == Y;
  }
  T X;
  T Y;
};

template <typename T>
bool operator==(const Point<T> &A, const Point<T> &B) {
  return A.X == B.X && A.Y == B.Y;
}

template <class T>
struct PointPlus {
  using P = Point<T>;
  P operator()(const P &A, const P &B) const {
    return P(A.X + B.X, A.Y + B.Y);
  }
};

template <typename SpecializationKernelName, typename T, int Dim, class BinaryOperation>
void testKnown(T Identity, T A, T B) {

  BinaryOperation BOp;
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
    test_reducer(Redu, Identity, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<SpecializationKernelName>([=]() {});
  });
}

template <typename SpecializationKernelName, typename T, int Dim, class BinaryOperation>
void testUnknown(T Identity, T A, T B) {

  BinaryOperation BOp;
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
    test_reducer(Redu, Identity, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<SpecializationKernelName>([=]() {});
  });
}

template <typename SpecializationKernelName, typename T, class BinaryOperation>
void testBoth(T Identity, T A, T B) {
  testKnown<KernelNameGroup<SpecializationKernelName, class KernelName_SpronAvHpacKFL>, T, 0, BinaryOperation>(Identity, A, B);
  testKnown<KernelNameGroup<SpecializationKernelName, class KernelName_XFxrYatPJlU>, T, 1, BinaryOperation>(Identity, A, B);
  testUnknown<KernelNameGroup<SpecializationKernelName, class KernelName_oUFYMyQSlL>, T, 0, BinaryOperation>(Identity, A, B);
  testUnknown<KernelNameGroup<SpecializationKernelName, class KernelName_Ndbp>, T, 1, BinaryOperation>(Identity, A, B);
}

int main() {
  // testKnown does not pass identity to reduction ctor.
  testBoth<class KernelName_DpWavJTNjhJtrHmLWt, int, intel::plus<int>>(0, 1, 7);
  testBoth<class KernelName_MHRtc, int, std::multiplies<int>>(1, 1, 7);
  testBoth<class KernelName_eYhurMyKBZvzctmqwUZ, int, intel::bit_or<int>>(0, 1, 8);
  testBoth<class KernelName_DpVPIUBjUMGZEwBFHH, int, intel::bit_xor<int>>(0, 7, 3);
  testBoth<class KernelName_vGKFactgrkngMXd, int, intel::bit_and<int>>(~0, 7, 3);
  testBoth<class KernelName_GLpknSBxclKWjm, int, intel::minimum<int>>((std::numeric_limits<int>::max)(), 7, 3);
  testBoth<class KernelName_EvOaOYQ, int, intel::maximum<int>>((std::numeric_limits<int>::min)(), 7, 3);

  testBoth<class KernelName_iFbcoTtPeDtUEK, float, intel::plus<float>>(0, 1, 7);
  testBoth<class KernelName_PEMJanstdNezDSXnP, float, std::multiplies<float>>(1, 1, 7);
  testBoth<class KernelName_wOEuftXSjCLpoTOMrYHR, float, intel::minimum<float>>(getMaximumFPValue<float>(), 7, 3);
  testBoth<class KernelName_HzFCIZQKeV, float, intel::maximum<float>>(getMinimumFPValue<float>(), 7, 3);

  testUnknown<class KernelName_sJOZPgFeiALyqwIWnFP, Point<float>, 0, PointPlus<float>>(Point<float>(0), Point<float>(1), Point<float>(7));
  testUnknown<class KernelName_jMA, Point<float>, 1, PointPlus<float>>(Point<float>(0), Point<float>(1), Point<float>(7));

  std::cout << "Test passed\n";
  return 0;
}
