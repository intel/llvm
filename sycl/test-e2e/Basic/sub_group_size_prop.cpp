// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <iostream>

using namespace sycl;

enum class Variant { Function, Functor, FunctorAndProperty };

template <Variant KernelVariant, size_t SGSize> class SubGroupKernel;

template <size_t SGSize> struct KernelFunctorWithSGSizeProp {
  accessor<size_t, 1, access_mode::write> Acc;

  KernelFunctorWithSGSizeProp(accessor<size_t, 1, access_mode::write> Acc)
      : Acc(Acc) {}

  void operator()(nd_item<1> NdItem) const {
    auto SG = NdItem.get_sub_group();
    if (NdItem.get_global_linear_id() == 0)
      Acc[0] = SG.get_local_linear_range();
  }

  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<SGSize>};
  }
};

template <size_t SGSize>
void test(queue &Queue, const std::vector<size_t> SupportedSGSizes) {
  std::cout << "Testing sub_group_size property for sub-group size=" << SGSize
            << std::endl;

  auto SGSizeSupported =
      std::find(SupportedSGSizes.begin(), SupportedSGSizes.end(), SGSize) !=
      SupportedSGSizes.end();
  if (!SGSizeSupported) {
    std::cout << "Sub-group size " << SGSize
              << " is not supported on the device." << std::endl;
    return;
  }

  auto Props = ext::oneapi::experimental::properties{
      ext::oneapi::experimental::sub_group_size<SGSize>};

  nd_range<1> NdRange(SGSize * 4, SGSize * 2);

  size_t ReadSubGroupSize = 0;
  {
    buffer ReadSubGroupSizeBuf(&ReadSubGroupSize, range(1));

    Queue.submit([&](handler &CGH) {
      accessor ReadSubGroupSizeBufAcc{ReadSubGroupSizeBuf, CGH,
                                      sycl::write_only, sycl::no_init};

      CGH.parallel_for<SubGroupKernel<Variant::Function, SGSize>>(
          NdRange, Props, [=](nd_item<1> NdItem) {
            auto SG = NdItem.get_sub_group();
            if (NdItem.get_global_linear_id() == 0)
              ReadSubGroupSizeBufAcc[0] = SG.get_local_linear_range();
          });
    });
  }
  assert(ReadSubGroupSize == SGSize && "Failed check for function.");

  ReadSubGroupSize = 0;
  {
    buffer ReadSubGroupSizeBuf(&ReadSubGroupSize, range(1));

    Queue.submit([&](handler &CGH) {
      accessor ReadSubGroupSizeBufAcc{ReadSubGroupSizeBuf, CGH,
                                      sycl::write_only, sycl::no_init};
      KernelFunctorWithSGSizeProp<SGSize> KernelFunctor{ReadSubGroupSizeBufAcc};

      CGH.parallel_for<SubGroupKernel<Variant::Functor, SGSize>>(NdRange,
                                                                 KernelFunctor);
    });
  }
  assert(ReadSubGroupSize == SGSize && "Failed check for functor.");

  ReadSubGroupSize = 0;
  {
    buffer ReadSubGroupSizeBuf(&ReadSubGroupSize, range(1));

    Queue.submit([&](handler &CGH) {
      accessor ReadSubGroupSizeBufAcc{ReadSubGroupSizeBuf, CGH,
                                      sycl::write_only, sycl::no_init};
      KernelFunctorWithSGSizeProp<SGSize> KernelFunctor{ReadSubGroupSizeBufAcc};

      CGH.parallel_for<SubGroupKernel<Variant::Functor, SGSize>>(NdRange, Props,
                                                                 KernelFunctor);
    });
  }
  assert(ReadSubGroupSize == SGSize &&
         "Failed check for functor and properties.");
}

int main() {
  queue Q;
  std::vector<size_t> SupportedSGSizes =
      Q.get_device().get_info<info::device::sub_group_sizes>();

  test<1>(Q, SupportedSGSizes);
  test<8>(Q, SupportedSGSizes);
  test<16>(Q, SupportedSGSizes);
  test<32>(Q, SupportedSGSizes);
  test<64>(Q, SupportedSGSizes);

  return 0;
}
