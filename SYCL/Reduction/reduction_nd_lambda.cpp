// Test disabled due to sporadic failure
// REQUIRES: TEMPORARILY_DISABLED
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Inconsistently fails on HIP AMD, error message `Barrier is not supported on
// the host device yet.` on HIP Nvidia.
// UNSUPPORTED: hip_amd || hip_nvidia

// This test performs basic checks of parallel_for(nd_range, reduction, lambda)

#include "reduction_nd_range_scalar.hpp"

using namespace cl::sycl;

template <typename T, access::mode M> class MName;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  constexpr access::mode DW = access::mode::discard_write;
  constexpr access::mode RW = access::mode::read_write;
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += testBoth<MName<Name, DW>, DW>(Q, Identity, Init, BOp, NDRange);
  NumErrors += testBoth<MName<Name, RW>, RW>(Q, Identity, Init, BOp, NDRange);
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
