// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// This performs basic checks such as reduction creation, getIdentity() method,
// and the combine() method of the aux class 'reducer'.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

bool toBool(bool V) { return V; }
bool toBool(vec<int, 2> V) { return V.x() && V.y(); }
bool toBool(vec<int, 4> V) { return V.x() && V.y() && V.z() && V.w(); }

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
  assert(toBool(ExpectedValue == Reducer.MValue) &&
         "Wrong result of binary operation.");
}

template <typename... Ts> class KernelNameGroup;

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void testKnown(T Identity, BinaryOperation BOp, T A, T B) {
  static_assert(has_known_identity<BinaryOperation, T>::value);
  queue Q;
  buffer<T, 1> ReduBuf(1);
  T* ReduUSMPtr = malloc_host<T>(1, Q);

  Q.submit([&](handler &CGH) {
    // Reduction needs a global_buffer accessor as a parameter.
    // This accessor is not really used in this test.
    accessor<T, Dim, access::mode::read_write, access::target::global_buffer>
        ReduRWAcc(ReduBuf, CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        ReduDWAcc(ReduBuf, CGH);
    auto ReduRW = ONEAPI::reduction(ReduRWAcc, BOp);
    auto ReduDW = ONEAPI::reduction(ReduDWAcc, BOp);
    auto ReduRWUSM = ONEAPI::reduction(ReduUSMPtr, BOp);
    auto ReduRW2020 = sycl::reduction(ReduBuf, CGH, BOp);
    auto ReduRWUSM2020 = sycl::reduction(ReduUSMPtr, BOp);

    assert(toBool(ReduRW.getIdentity() == Identity) &&
           toBool(ReduDW.getIdentity() == Identity) &&
           toBool(ReduRWUSM.getIdentity() == Identity) &&
           toBool(ReduRW2020.getIdentity() == Identity) &&
           toBool(ReduRWUSM2020.getIdentity() == Identity) &&
           toBool(known_identity<BinaryOperation, T>::value == Identity) &&
           "Failed getIdentity() check().");
    test_reducer(ReduRW, A, B);
    test_reducer(ReduDW, A, B);
    test_reducer(ReduRWUSM, A, B);
    test_reducer(ReduRW2020, A, B);
    test_reducer(ReduRWUSM2020, A, B);

    test_reducer(ReduRW, Identity, BOp, A, B);
    test_reducer(ReduDW, Identity, BOp, A, B);
    test_reducer(ReduRWUSM, Identity, BOp, A, B);
    test_reducer(ReduRW2020, Identity, BOp, A, B);
    test_reducer(ReduRWUSM2020, Identity, BOp, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<SpecializationKernelName>([=]() {});
  });
  free(ReduUSMPtr, Q);
}

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void testUnknown(T Identity, BinaryOperation BOp, T A, T B) {
  queue Q;
  buffer<T, 1> ReduBuf(1);
  T* ReduUSMPtr = malloc_host<T>(1, Q);
  Q.submit([&](handler &CGH) {
    // Reduction needs a global_buffer accessor as a parameter.
    // This accessor is not really used in this test.
    accessor<T, Dim, access::mode::read_write, access::target::global_buffer>
        ReduRWAcc(ReduBuf, CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        ReduDWAcc(ReduBuf, CGH);
    auto ReduRW = ONEAPI::reduction(ReduRWAcc, Identity, BOp);
    auto ReduDW = ONEAPI::reduction(ReduDWAcc, Identity, BOp);
    auto ReduRWUSM = ONEAPI::reduction(ReduUSMPtr, Identity, BOp);
    auto ReduRW2020 = sycl::reduction(ReduBuf, CGH, Identity, BOp);
    auto ReduRWUSM2020 = sycl::reduction(ReduUSMPtr, Identity, BOp);
    assert(toBool(ReduRW.getIdentity() == Identity) &&
           toBool(ReduDW.getIdentity() == Identity) &&
           toBool(ReduRWUSM.getIdentity() == Identity) &&
           toBool(ReduRW2020.getIdentity() == Identity) &&
           toBool(ReduRWUSM2020.getIdentity() == Identity) &&
           "Failed getIdentity() check().");
    test_reducer(ReduRW, Identity, BOp, A, B);
    test_reducer(ReduDW, Identity, BOp, A, B);
    test_reducer(ReduRWUSM, Identity, BOp, A, B);
    test_reducer(ReduRW2020, Identity, BOp, A, B);
    test_reducer(ReduRWUSM2020, Identity, BOp, A, B);

    // Command group must have at least one task in it. Use an empty one.
    CGH.single_task<SpecializationKernelName>([=]() {});
  });
  free(ReduUSMPtr, Q);
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

  int2 IdentityI2 = {0, 0};
  int2 AI2 = {1, 2};
  int2 BI2 = {7, 13};
  testUnknown<class KNI2, int2, 0>(IdentityI2, ONEAPI::plus<int2>(), AI2, BI2);

  float4 IdentityF4 = {0, 0, 0, 0};
  float4 AF4 = {1, 2, -1, -34};
  float4 BF4 = {7, 13, 0, 35};
  testUnknown<class KNF4, float4, 0>(IdentityF4, ONEAPI::plus<>(), AF4, BF4);

  std::cout << "Test passed\n";
  return 0;
}
