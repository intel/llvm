// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with types that may require additional runtime checks for extensions
// supported by the device, e.g. 'half' or 'double'

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename SpecializationKernelName, typename T, int Dim,
          access::mode Mode, class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  if (Mode == access::mode::read_write)
    (OutBuf.template get_access<access::mode::write>())[0] = Identity;

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, Mode, access::target::global_buffer> Out(OutBuf, CGH);
    auto Redu = ONEAPI::reduction(Out, Identity, BOp);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });

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
    assert(0 && "Wrong value.");
  }
}

template <typename T> int runTests(const string_class &ExtensionName) {
  device D = default_selector().select_device();
  if (!D.is_host() && !D.has_extension(ExtensionName)) {
    std::cout << "Test skipped\n";
    return 0;
  }

  // Check some less standards WG sizes and corner cases first.
  test<class KernelName_oTh, T, 1, access::mode::read_write,
       std::multiplies<T>>(0, 4, 4);
  test<class KernelName_QUQnMARQT, T, 0, access::mode::discard_write,
       ONEAPI::plus<T>>(0, 4, 64);

  test<class KernelName_xGixNo, T, 0, access::mode::read_write,
       ONEAPI::minimum<T>>(getMaximumFPValue<T>(), 7, 7);
  test<class KernelName_qXNFw, T, 1, access::mode::discard_write,
       ONEAPI::maximum<T>>(getMinimumFPValue<T>(), 7, 7 * 5);

#if __cplusplus >= 201402L
  test<class KernelName_lXdWtzANdDcvm, T, 1, access::mode::read_write,
       ONEAPI::plus<>>(1, 3, 3 * 5);
  test<class KernelName_FDQalsDxmbi, T, 1, access::mode::discard_write,
       ONEAPI::minimum<>>(getMaximumFPValue<T>(), 3, 3);
  test<class KernelName_TaNRRxDRXbzYrFImPYC, T, 0, access::mode::discard_write,
       ONEAPI::maximum<>>(getMinimumFPValue<T>(), 3, 3);
#endif // __cplusplus >= 201402L

  std::cout << "Test passed\n";
  return 0;
}
