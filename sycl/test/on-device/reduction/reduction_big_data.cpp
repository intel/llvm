// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// RUNx: %RUN_ON_HOST %t.out
// TODO: Enable the test for HOST when it supports ONEAPI::reduce() and
// barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// where the bigger data size and/or non-uniform work-group sizes may cause
// errors.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>

using namespace cl::sycl;

template <typename... Ts> class KernelNameGroup;

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        Out(OutBuf, CGH);
    size_t NWorkGroups = (NWItems - 1) / WGSize + 1;
    range<1> GlobalRange(NWorkGroups * WGSize);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    std::cout << "Running the test with: GlobalRange = "
              << (NWorkGroups * WGSize) << ", LocalRange = " << WGSize
              << ", NWItems = " << NWItems << "\n";
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, ONEAPI::reduction(Out, Identity, BOp),
        [=](nd_item<1> NDIt, auto &Sum) {
          if (NDIt.get_global_linear_id() < NWItems)
            Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

template <typename T> struct BigCustomVec : public CustomVec<T> {
  BigCustomVec() : CustomVec<T>() {}
  BigCustomVec(T X, T Y) : CustomVec<T>(X, Y) {}
  BigCustomVec(T V) : CustomVec<T>(V) {}
  unsigned char OtherData[512 - sizeof(CustomVec<T>)];
};

template <class T> struct BigCustomVecPlus {
  using CV = BigCustomVec<T>;
  CV operator()(const CV &A, const CV &B) const {
    return CV(A.X + B.X, A.Y + B.Y);
  }
};

int main() {
  device Device = queue().get_device();
  std::size_t MaxWGSize = Device.get_info<info::device::max_work_group_size>();
  std::size_t LocalMemSize = Device.get_info<info::device::local_mem_size>();
  std::cout << "Detected device::max_work_group_size = " << MaxWGSize << "\n";
  std::cout << "Detected device::local_mem_size = " << LocalMemSize << "\n";

  test<class KernelName_slumazIfW, float, 0, ONEAPI::maximum<>>(
      getMinimumFPValue<float>(), MaxWGSize / 2, MaxWGSize * MaxWGSize + 1);

  size_t MaxUsableWGSize = LocalMemSize / sizeof(BigCustomVec<long long>);
  if ((MaxUsableWGSize & (MaxUsableWGSize - 1)) != 0)
    MaxUsableWGSize--;// Need 1 additional element in local mem if not pow of 2
  size_t UsableWGSize = std::min(MaxUsableWGSize / 2, MaxWGSize);
  test<class KernelName_VzSVAWkAmHq, BigCustomVec<long long>, 1,
       BigCustomVecPlus<long long>>(BigCustomVec<long long>(0), UsableWGSize,
                                    UsableWGSize * MaxWGSize + 1);

  std::cout << "Test passed\n";
  return 0;
}
