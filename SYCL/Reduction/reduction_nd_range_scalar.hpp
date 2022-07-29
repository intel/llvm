// This test performs basic checks of parallel_for(range<Dims>, reduction, func)
// with reductions initialized with 1-dimensional buffer/accessor
// accessing a scalar holding the reduction result.

#include "reduction_utils.hpp"

using namespace sycl;

template <typename T, bool B> class KName;

template <typename Name, bool IsSYCL2020, access::mode Mode, int AccDim = 1,
          typename T, class BinaryOperation, int Dims>
int test(queue &Q, T Identity, T Init, BinaryOperation BOp,
         const nd_range<Dims> &Range) {
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
  range<Dims> GlobalRange = Range.get_global_range();
  if (GlobalRange.size() * sizeof(T) > OneGB) {
    std::cout << " SKIPPED due to too big data size" << std::endl;
    return 0;
  }

  // TODO: Perhaps, this is a _temporary_ fix for CI. The test may run
  // for too long when the range is big. That is especially bad on ACC.
  if (GlobalRange.size() > 65536 && Q.get_device().is_accelerator()) {
    std::cout << " SKIPPED due to risk of timeout in CI" << std::endl;
    return 0;
  }

  buffer<T, Dims> InBuf(GlobalRange);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, GlobalRange);
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
    CGH.parallel_for<Name>(Range, Redu, [=](nd_item<Dims> NDIt, auto &Sum) {
      Sum.combine(In[NDIt.get_global_id()]);
    });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  return checkResults(Q, IsSYCL2020, BOp, Range, ComputedOut, CorrectOut);
}

template <typename Name, access::mode Mode, typename T, class BinaryOperation,
          int Dims>
int testBoth(queue &Q, T Identity, T Init, BinaryOperation BOp,
             const nd_range<Dims> &Range) {
  return test<KName<Name, false>, false, Mode>(Q, Identity, Init, BOp, Range) +
         test<KName<Name, true>, true, Mode>(Q, Identity, Init, BOp, Range);
}
