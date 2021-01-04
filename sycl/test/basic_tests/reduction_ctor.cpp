// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

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

template <typename... Ts> class KernelNameGroup;

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void testKnown(T Identity, BinaryOperation BOp, T A, T B) {
  buffer<T, 1> ReduBuf(1);

  queue Q;
  Q.submit([&](handler &CGH) {
    // Reduction needs a global_buffer accessor as a parameter.
    // This accessor is not really used in this test.
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        ReduAcc(ReduBuf, CGH);
    auto Redu = ONEAPI::reduction(ReduAcc, BOp);
    assert(Redu.getIdentity() == Identity && "Failed getIdentity() check().");
    test_reducer(Redu, A, B);
    test_reducer(Redu, Identity, BOp, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<SpecializationKernelName>([=]() {});
  });
}

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void testUnknown(T Identity, BinaryOperation BOp, T A, T B) {
  buffer<T, 1> ReduBuf(1);
  queue Q;
  Q.submit([&](handler &CGH) {
    // Reduction needs a global_buffer accessor as a parameter.
    // This accessor is not really used in this test.
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        ReduAcc(ReduBuf, CGH);
    auto Redu = ONEAPI::reduction(ReduAcc, Identity, BOp);
    assert(Redu.getIdentity() == Identity && "Failed getIdentity() check().");
    test_reducer(Redu, Identity, BOp, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<SpecializationKernelName>([=]() {});
  });
}

template <typename SpecializationKernelName, typename T, class BinaryOperation>
void testBoth(T Identity, BinaryOperation BOp, T A, T B) {
  testKnown<KernelNameGroup<SpecializationKernelName,
                            class KernelName_SpronAvHpacKFL>,
            T, 0>(Identity, BOp, A, B);
  testKnown<
      KernelNameGroup<SpecializationKernelName, class KernelName_XFxrYatPJlU>,
      T, 1>(Identity, BOp, A, B);
  testUnknown<
      KernelNameGroup<SpecializationKernelName, class KernelName_oUFYMyQSlL>, T,
      0>(Identity, BOp, A, B);
  testUnknown<KernelNameGroup<SpecializationKernelName, class KernelName_Ndbp>,
              T, 1>(Identity, BOp, A, B);
}

int main() {
  testBoth<class KernelName_DpWavJTNjhJtrHmLWt, int>(0, ONEAPI::plus<int>(), 1,
                                                     7);
  testBoth<class KernelName_MHRtc, int>(1, std::multiplies<int>(), 1, 7);
  testBoth<class KernelName_eYhurMyKBZvzctmqwUZ, int>(0, ONEAPI::bit_or<int>(),
                                                      1, 8);
  testBoth<class KernelName_DpVPIUBjUMGZEwBFHH, int>(0, ONEAPI::bit_xor<int>(),
                                                     7, 3);
  testBoth<class KernelName_vGKFactgrkngMXd, int>(~0, ONEAPI::bit_and<int>(), 7,
                                                  3);
  testBoth<class KernelName_GLpknSBxclKWjm, int>(
      (std::numeric_limits<int>::max)(), ONEAPI::minimum<int>(), 7, 3);
  testBoth<class KernelName_EvOaOYQ, int>((std::numeric_limits<int>::min)(),
                                          ONEAPI::maximum<int>(), 7, 3);

  testBoth<class KernelName_iFbcoTtPeDtUEK, float>(0, ONEAPI::plus<float>(), 1,
                                                   7);
  testBoth<class KernelName_PEMJanstdNezDSXnP, float>(
      1, std::multiplies<float>(), 1, 7);
  testBoth<class KernelName_wOEuftXSjCLpoTOMrYHR, float>(
      getMaximumFPValue<float>(), ONEAPI::minimum<float>(), 7, 3);
  testBoth<class KernelName_HzFCIZQKeV, float>(getMinimumFPValue<float>(),
                                               ONEAPI::maximum<float>(), 7, 3);

  testUnknown<class KernelName_sJOZPgFeiALyqwIWnFP, CustomVec<float>, 0,
              CustomVecPlus<float>>(CustomVec<float>(0), CustomVecPlus<float>(),
                                    CustomVec<float>(1), CustomVec<float>(7));
  testUnknown<class KernelName_jMA, CustomVec<float>, 1>(
      CustomVec<float>(0), CustomVecPlus<float>(), CustomVec<float>(1),
      CustomVec<float>(7));

  testUnknown<class KernelName_zhF, int, 0>(
      0, [](auto a, auto b) { return a | b; }, 1, 8);

  std::cout << "Test passed\n";
  return 0;
}
