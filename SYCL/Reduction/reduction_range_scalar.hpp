// This test performs basic checks of parallel_for(range<Dims>, reduction, func)
// with reductions initialized with 1-dimensional buffer/accessor
// accessing a scalar holding the reduction result.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T, bool B> class KName;

template <int Dims>
std::ostream &operator<<(std::ostream &OS, const range<Dims> &Range) {
  OS << "{" << Range[0];
  if constexpr (Dims > 1)
    OS << ", " << Range[1];
  if constexpr (Dims > 2)
    OS << ", " << Range[2];
  OS << "}";
  return OS;
}

template <typename Name, bool IsSYCL2020Mode, access::mode Mode, typename T,
          int AccDim = 1, class BinaryOperation, int Dims>
void test(queue &Q, T Identity, T Init, BinaryOperation BOp,
          range<Dims> Range) {
  std::string StdMode = IsSYCL2020Mode ? "SYCL2020" : "ONEAPI  ";
  std::cout << "Running the test case: " << StdMode
            << " {T=" << typeid(T).name()
            << ", BOp=" << typeid(BinaryOperation).name() << ", Range=" << Range
            << std::endl;

  // Skip the test for such big arrays now.
  constexpr size_t TwoGB = 2LL * 1024 * 1024 * 1024;
  if (Range.size() > TwoGB)
    return;

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
  if constexpr (IsSYCL2020Mode) {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      property_list PropList = getPropertyList<Mode>();
      auto Redu = sycl::reduction(OutBuf, CGH, Identity, BOp, PropList);

      CGH.parallel_for<Name>(
          Range, Redu, [=](id<Dims> Id, auto &Sum) { Sum.combine(In[Id]); });
    });
  } else {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      accessor<T, AccDim, Mode, access::target::global_buffer> Out(OutBuf, CGH);
      auto Redu = ext::oneapi::reduction(Out, Identity, BOp);

      CGH.parallel_for<Name>(
          Range, Redu, [=](id<Dims> Id, auto &Sum) { Sum.combine(In[Id]); });
    });
  }

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    printDeviceInfo(Q, true);
    std::cerr << "Error: Range = " << Range << ", "
              << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

template <typename Name, access::mode Mode, typename T, class BinaryOperation,
          int Dims>
void testBoth(queue &Q, T Identity, T Init, BinaryOperation BOp,
              range<Dims> Range) {
  test<KName<Name, false>, false, Mode>(Q, Identity, Init, BOp, Range);
  test<KName<Name, true>, true, Mode>(Q, Identity, Init, BOp, Range);
}
