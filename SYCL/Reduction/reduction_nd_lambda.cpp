// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, lambda)

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

// Note that this function is created only to test that if the accessor
// object passed to ONEAPI::reduction is destroyed right after
// ONEAPI::reduction creation, then the reduction still works properly,
// i.e. it holds a COPY of user's accessor.
template <typename T, typename BOpT>
auto createReduction(sycl::buffer<T, 1> Buffer, handler &CGH, T Identity,
                     BOpT BOp) {
  auto Acc = Buffer.template get_access<access::mode::discard_write>(CGH);
  return ONEAPI::reduction(Acc, Identity, BOp);
}

template <class KernelName, typename T, class BinaryOperation>
void test(queue &Q, T Identity, BinaryOperation BOp, size_t WGSize,
          size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    auto Redu = createReduction(OutBuf, CGH, Identity, BOp);

    nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
    CGH.parallel_for<KernelName>(NDRange, Redu,
                                 [=](nd_item<1> NDIt, auto &Sum) {
                                   Sum.combine(In[NDIt.get_global_linear_id()]);
                                 });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cerr << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cerr << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

int main() {
  queue Q;
  test<class AddTestName, int>(
      Q, 0, [](auto x, auto y) { return (x + y); }, 1, 1024);
  test<class MulTestName, int>(
      Q, 0, [](auto x, auto y) { return (x * y); }, 8, 32);

  // Check with CUSTOM type.
  test<class CustomAddTestname, CustomVec<long long>>(
      Q, CustomVec<long long>(0),
      [](auto x, auto y) {
        CustomVecPlus<long long> BOp;
        return BOp(x, y);
      },
      4, 64);

  std::cout << "Test passed\n";
  return 0;
}
