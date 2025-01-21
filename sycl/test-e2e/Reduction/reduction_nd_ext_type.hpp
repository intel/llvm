// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with types that may require additional runtime checks for extensions
// supported by the device, e.g. 'half' or 'double'

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation,
          typename PropListTy = property_list>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems, PropListTy PropList = {}) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += test<Name>(Q, Identity, Init, BOp, NDRange, PropList);
}

template <typename T> int runTests(sycl::aspect ExtAspect) {
  queue Q;
  printDeviceInfo(Q);
  device D = Q.get_device();
  if (!D.has(ExtAspect)) {
    std::cout << "Test skipped\n";
    return 0;
  }

  constexpr access::mode RW = access::mode::read_write;
  constexpr access::mode DW = access::mode::discard_write;

  tests<class A1, T>(Q, 1, 77, std::multiplies<T>{}, 4, 4, init_to_identity());
  tests<class A2, T>(Q, 1, 77, std::multiplies<T>{}, 4, 8);

  tests<class B1, T>(Q, 0, 77, std::plus<T>{}, 4, 32, init_to_identity());
  tests<class B2, T>(Q, 0, 33, std::plus<T>{}, 3, 3 * 5);

  tests<class C1, T>(Q, getMaximumFPValue<T>(), -10.0,
                     ext::oneapi::minimum<T>{}, 7, 7 * 512, init_to_identity());
  tests<class C2, T>(Q, getMaximumFPValue<T>(), 99.0, ext::oneapi::minimum<T>{},
                     7, 7);

  tests<class D1, T>(Q, getMinimumFPValue<T>(), 99.0, ext::oneapi::maximum<>{},
                     3, 3, init_to_identity());
  tests<class D2, T>(Q, getMinimumFPValue<T>(), 99.0, ext::oneapi::maximum<>{},
                     7, 7 * 5);

  printFinalStatus(NumErrors);
  return NumErrors;
}
