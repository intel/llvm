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
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  if constexpr (IsSYCL2020Mode) {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      auto Redu = sycl::reduction(OutBuf, CGH, Identity, BOp);

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
    assert(0 && "Wrong value.");
  }
}

template <typename Name, typename T, int Dim, access::mode Mode,
          class BinaryOperation>
void testBoth(T Identity, size_t WGSize, size_t NWItems) {
  test<KName<Name, false>, false, T, Dim, Mode, BinaryOperation>(
      Identity, WGSize, NWItems);

  // TODO: property::reduction::initialize_to_identity is not supported yet.
  // Thus only read_write mode is tested now.
  constexpr access::mode _Mode = (Mode == DW) ? RW : Mode;
  test<KName<Name, true>, true, T, Dim, _Mode, BinaryOperation>(
      Identity, WGSize, NWItems);
}

template <typename T> int runTests(const string_class &ExtensionName) {
  device D = default_selector().select_device();
  if (!D.is_host() && !D.has_extension(ExtensionName)) {
    std::cout << "Test skipped\n";
    return 0;
  }

  // Check some less standards WG sizes and corner cases first.
  testBoth<class A, T, 1, RW, std::multiplies<T>>(0, 4, 4);
  testBoth<class B, T, 0, DW, ONEAPI::plus<T>>(0, 4, 64);

  testBoth<class C, T, 0, RW, ONEAPI::minimum<T>>(getMaximumFPValue<T>(), 7, 7);
  testBoth<class D, T, 1, access::mode::discard_write, ONEAPI::maximum<T>>(
      getMinimumFPValue<T>(), 7, 7 * 5);

  testBoth<class E, T, 1, RW, ONEAPI::plus<>>(1, 3, 3 * 5);
  testBoth<class F, T, 1, DW, ONEAPI::minimum<>>(getMaximumFPValue<T>(), 3, 3);
  testBoth<class G, T, 0, DW, ONEAPI::maximum<>>(getMinimumFPValue<T>(), 3, 3);

  std::cout << "Test passed\n";
  return 0;
}
