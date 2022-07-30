// This test performs basic checks of parallel_for(range<Dims>, reduction, func)
// with reductions initialized with 1-dimensional buffer/accessor
// accessing a scalar holding the reduction result.

#include "reduction_utils.hpp"
#include <iostream>

using namespace sycl;

template <typename T, bool B> class KName;
template <typename T, typename> class TName;

template <typename Name, bool IsSYCL2020, access::mode Mode, int AccDim = 1,
          typename T, class BinaryOperation, int Dims>
int test(queue &Q, T Identity, T Init, BinaryOperation BOp,
         const range<Dims> &Range) {
  printTestLabel<T, BinaryOperation>(IsSYCL2020, Range);

  // It is a known problem with passing data that is close to 4Gb in size
  // to device. Such data breaks the execution pretty badly.
  // Some of test cases calling this function try to verify the correctness
  // of reduction with the global range bigger than the maximal work-group size
  // for the device. Maximal WG size for device may be very big, e.g. it is
  // 67108864 for ACC emulator. Multiplying that by some factor
  // (to exceed max WG-Size) and multiplying it by the element size may exceed
  // the safe size of data passed to device.
  // Let's set it to 1 GB for now, and just skip the test if it exceeds 1Gb.
  constexpr size_t OneGB = 1LL * 1024 * 1024 * 1024;
  if (Range.size() * sizeof(T) > OneGB) {
    std::cout << " SKIPPED due to too big data size" << std::endl;
    return 0;
  }

  // TODO: Perhaps, this is a _temporary_ fix for CI. The test may run
  // for too long when the range is big. That is especially bad on ACC.
  if (Range.size() > 65536 && Q.get_device().is_accelerator()) {
    std::cout << " SKIPPED due to risk of timeout in CI" << std::endl;
    return 0;
  }

  buffer<T, Dims> InBuf(Range);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, Range);
  if constexpr (Mode == access::mode::read_write) {
    CorrectOut = BOp(CorrectOut, Init);
  }

  // The value assigned here must be discarded (if IsReadWrite is true).
  // Verify that it is really discarded and assign some value.
  (OutBuf.template get_access<access::mode::write>())[0] = Init;

  // Compute.
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    auto Redu =
        createReduction<IsSYCL2020, Mode, AccDim>(OutBuf, CGH, Identity, BOp);
    CGH.parallel_for<Name>(
        Range, Redu, [=](id<Dims> Id, auto &Sum) { Sum.combine(In[Id]); });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  return checkResults(Q, IsSYCL2020, BOp, Range, ComputedOut, CorrectOut);
}

template <typename Name, access::mode Mode, typename T, class BinaryOperation,
          int Dims>
int testBoth(queue &Q, T Identity, T Init, BinaryOperation BOp,
             const range<Dims> &Range) {
  return test<KName<Name, false>, false, Mode>(Q, Identity, Init, BOp, Range) +
         test<KName<Name, true>, true, Mode>(Q, Identity, Init, BOp, Range);
}

template <typename Name, bool IsSYCL2020, access::mode Mode, typename T,
          class BinaryOperation, int Dims>
int testUSM(queue &Q, T Identity, T Init, BinaryOperation BOp,
            const range<Dims> &Range, usm::alloc AllocType) {
  printTestLabel<T, BinaryOperation>(IsSYCL2020, Range);

  auto Dev = Q.get_device();
  if (!Dev.has(getUSMAspect(AllocType))) {
    std::cout << " SKIPPED due to unsupported USM alloc type" << std::endl;
    return 0;
  }

  // It is a known problem with passing data that is close to 4Gb in size
  // to device. Such data breaks the execution pretty badly.
  // Some of test cases calling this function try to verify the correctness
  // of reduction with the global range bigger than the maximal work-group size
  // for the device. Maximal WG size for device may be very big, e.g. it is
  // 67108864 for ACC emulator. Multiplying that by some factor
  // (to exceed max WG-Size) and multiplying it by the element size may exceed
  // the safe size of data passed to device.
  // Let's set it to 1 GB for now, and just skip the test if it exceeds 1Gb.
  constexpr size_t OneGB = 1LL * 1024 * 1024 * 1024;
  if (Range.size() * sizeof(T) > OneGB) {
    std::cout << " SKIPPED due to too big data size" << std::endl;
    return 0;
  }

  // TODO: Perhaps, this is a _temporary_ fix for CI. The test may run
  // for too long when the range is big. That is especially bad on ACC.
  if (Range.size() > 65536) {
    std::cout << " SKIPPED due to risk of timeout in CI" << std::endl;
    return 0;
  }

  T *ReduVarPtr = (T *)malloc(sizeof(T), Dev, Q.get_context(), AllocType);
  if (ReduVarPtr == nullptr) {
    std::cout << " SKIPPED due to unrelated reason: alloc returned nullptr"
              << std::endl;
    return 0;
  }
  if (AllocType == usm::alloc::device) {
    Q.submit([&](handler &CGH) {
       CGH.single_task<TName<Name, class InitKernel>>(
           [=]() { *ReduVarPtr = Init; });
     }).wait();
  } else {
    *ReduVarPtr = Init;
  }

  // Initialize.
  T CorrectOut;
  buffer<T, Dims> InBuf(Range);
  initInputData(InBuf, CorrectOut, Identity, BOp, Range);
  if constexpr (Mode == access::mode::read_write)
    CorrectOut = BOp(CorrectOut, Init);

  // Compute.
  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Redu = createReduction<IsSYCL2020, Mode>(ReduVarPtr, Identity, BOp);
     CGH.parallel_for<TName<Name, class Test>>(
         Range, Redu, [=](id<Dims> Id, auto &Sum) { Sum.combine(In[Id]); });
   }).wait();

  // Check correctness.
  T ComputedOut;
  if (AllocType == usm::alloc::device) {
    buffer<T, 1> Buf(&ComputedOut, range<1>(1));
    Q.submit([&](handler &CGH) {
       auto OutAcc = Buf.template get_access<access::mode::discard_write>(CGH);
       CGH.single_task<TName<Name, class Check>>(
           [=]() { OutAcc[0] = *ReduVarPtr; });
     }).wait();
    ComputedOut = (Buf.template get_access<access::mode::read>())[0];
  } else {
    ComputedOut = *ReduVarPtr;
  }

  std::string AllocStr =
      "AllocMode=" + std::to_string(static_cast<int>(AllocType));
  int Error = checkResults(Q, IsSYCL2020, BOp, Range, ComputedOut, CorrectOut,
                           AllocStr);
  free(ReduVarPtr, Q.get_context());
  return Error;
}

template <typename Name, access::mode Mode, typename T, class BinaryOperation,
          int Dims>
int test2020USM(queue &Q, T Identity, T Init, BinaryOperation BOp,
                const range<Dims> &Range) {
  int NumErrors = 0;
  NumErrors += testUSM<TName<Name, class Shared2020>, true, Mode, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::shared);
  NumErrors += testUSM<TName<Name, class Host2020>, true, Mode, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::host);
  NumErrors += testUSM<TName<Name, class Device2020>, true, Mode, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::device);
  return NumErrors;
}

template <typename Name, access::mode Mode, typename T, class BinaryOperation,
          int Dims>
int testONEAPIUSM(queue &Q, T Identity, T Init, BinaryOperation BOp,
                  const range<Dims> &Range) {
  int NumErrors = 0;
  if (Mode == access::mode::discard_write) {
    std::cerr << "Skipped an incorrect test case: ext::oneapi::reduction "
              << "does not support discard_write mode for USM variables.";
    return 0;
  }
  NumErrors += testUSM<TName<Name, class Shared>, false, Mode, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::shared);
  NumErrors += testUSM<TName<Name, class Host>, false, Mode, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::host);
  NumErrors += testUSM<TName<Name, class Device>, false, Mode, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::device);
  return NumErrors;
}
