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

size_t getSafeMaxWGSize(size_t MaxWGSize, size_t MemSize, size_t OneElemSize) {
  size_t MaxNumElems = MemSize / OneElemSize;
  if ((MaxNumElems & (MaxNumElems - 1)) != 0)
    MaxNumElems--; // Need 1 additional element in mem if not pow of 2
  return std::min(MaxNumElems / 2, MaxWGSize);
}

template <typename KernelName, typename T, int Dim, class BinaryOperation>
void test(T Identity) {
  queue Q;
  device Device = Q.get_device();

  std::size_t MaxWGSize = Device.get_info<info::device::max_work_group_size>();
  std::size_t LocalMemSize = Device.get_info<info::device::local_mem_size>();
  std::cout << "Detected device::max_work_group_size = " << MaxWGSize << "\n";
  std::cout << "Detected device::local_mem_size = " << LocalMemSize << "\n";

  size_t WGSize = getSafeMaxWGSize(MaxWGSize, LocalMemSize, sizeof(T));

  size_t MaxGlobalMem = 2LL * 1024 * 1024 * 1024; // Don't use more than 2 Gb
  // Limit max global range by mem and also subtract 1 to make it non-uniform.
  size_t MaxGlobalRange = MaxGlobalMem / sizeof(T) - 1;
  size_t NWorkItems = std::min(WGSize * MaxWGSize + 1, MaxGlobalRange);

  size_t NWorkGroups = (NWorkItems - 1) / WGSize + 1;
  range<1> GlobalRange(NWorkGroups * WGSize);
  range<1> LocalRange(WGSize);
  nd_range<1> NDRange(GlobalRange, LocalRange);
  std::cout << "Running the test with: GlobalRange = " << (NWorkGroups * WGSize)
            << ", LocalRange = " << WGSize << ", NWorkItems = " << NWorkItems
            << "\n";

  buffer<T, 1> InBuf(NWorkItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWorkItems);

  // Compute.
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        Out(OutBuf, CGH);
    CGH.parallel_for<KernelName>(NDRange, ONEAPI::reduction(Out, Identity, BOp),
                                 [=](nd_item<1> NDIt, auto &Sum) {
                                   if (NDIt.get_global_linear_id() < NWorkItems)
                                     Sum.combine(
                                         In[NDIt.get_global_linear_id()]);
                                 });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
  std::cout << "Test case passed\n\n";
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
  test<class Test1, float, 0, ONEAPI::maximum<>>(getMinimumFPValue<float>());

  using BCV = BigCustomVec<long long>;
  test<class Test2, BCV, 1, BigCustomVecPlus<long long>>(BCV(0));

  std::cout << "Test passed\n";
  return 0;
}
