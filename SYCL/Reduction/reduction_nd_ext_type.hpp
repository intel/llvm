// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with types that may require additional runtime checks for extensions
// supported by the device, e.g. 'half' or 'double'

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T, bool B> class KName;
constexpr access::mode RW = access::mode::read_write;
constexpr access::mode DW = access::mode::discard_write;

template <typename Name, bool IsSYCL2020Mode, typename T, int Dim,
          access::mode Mode, class BinaryOperation>
void test(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);
  if (Mode == access::mode::read_write)
    CorrectOut = BOp(CorrectOut, Init);

  (OutBuf.template get_access<access::mode::write>())[0] = Init;

  // Compute.
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  if constexpr (IsSYCL2020Mode) {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      auto Redu =
          sycl::reduction(OutBuf, CGH, Identity, BOp, getPropertyList<Mode>());
      CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
        Sum.combine(In[NDIt.get_global_linear_id()]);
      });
    });
  } else {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      accessor<T, Dim, Mode, access::target::global_buffer> Out(OutBuf, CGH);
      auto Redu = ONEAPI::reduction(Out, Identity, BOp);

      CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
        Sum.combine(In[NDIt.get_global_linear_id()]);
      });
    });
  }

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  T MaxDiff = 3 * std::numeric_limits<T>::epsilon() *
              std::fabs(ComputedOut + CorrectOut);
  if (std::fabs(static_cast<T>(ComputedOut - CorrectOut)) > MaxDiff) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << ", MaxDiff = " << MaxDiff
              << "\n";
    if (IsSYCL2020Mode)
      std::cout << std::endl;
    assert(0 && "Wrong value.");
  }
}

template <typename Name, typename T, int Dim, access::mode Mode,
          class BinaryOperation>
void testBoth(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  test<KName<Name, false>, false, T, Dim, Mode, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems);

  test<KName<Name, true>, true, T, Dim, Mode, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems);
}

template <typename T> int runTests(const string_class &ExtensionName) {
  queue Q;
  device D = Q.get_device();
  if (!D.is_host() && !D.has_extension(ExtensionName)) {
    std::cout << "Test skipped\n";
    return 0;
  }

  testBoth<class A, T, 1, RW, std::multiplies<T>>(Q, 1, 77, 4, 4);

  testBoth<class B1, T, 0, DW, ONEAPI::plus<T>>(Q, 0, 77, 4, 64);
  testBoth<class B2, T, 1, RW, ONEAPI::plus<>>(Q, 0, 33, 3, 3 * 5);

  testBoth<class C1, T, 0, RW, ONEAPI::minimum<T>>(Q, getMaximumFPValue<T>(),
                                                   -10.0, 7, 7);
  testBoth<class C2, T, 0, RW, ONEAPI::minimum<T>>(Q, getMaximumFPValue<T>(),
                                                   99.0, 7, 7);
  testBoth<class C3, T, 1, DW, ONEAPI::minimum<>>(Q, getMaximumFPValue<T>(),
                                                  -99.0, 3, 3);

  testBoth<class D1, T, 0, DW, ONEAPI::maximum<>>(Q, getMinimumFPValue<T>(),
                                                  99.0, 3, 3);
  testBoth<class D2, T, 1, RW, ONEAPI::maximum<T>>(Q, getMinimumFPValue<T>(),
                                                   99.0, 7, 7 * 5);
  std::cout << "Test passed\n";
  return 0;
}
