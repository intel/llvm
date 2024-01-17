// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

// This test performs basic checks of parallel_for(nd_range, reduction, lambda)

#include "reduction_utils.hpp"

using namespace sycl;

template <typename T, access::mode M> class MName;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += test<KName<Name, true>>(Q, Identity, Init, BOp, NDRange);
  NumErrors += test<KName<Name, false>>(Q, Identity, Init, BOp, NDRange,
                                        init_to_identity());
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  tests<class A1, int>(
      Q, 0, 9, [](auto x, auto y) { return (x + y); }, 1, 1024);
  tests<class A2, uint64_t>(
      Q, 1, 2, [](auto x, auto y) { return (x * y); }, 8, 16);

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  tests<class A3, CV>(
      Q, CV(0), CV(2021),
      [](auto X, auto Y) {
        CustomVecPlus<long long> BOp;
        return BOp(X, Y);
      },
      4, 64);

  printFinalStatus(NumErrors);
  return NumErrors;
}
