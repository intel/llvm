#include "declarations.hpp"

template <typename T1, typename T2> struct KernelFunctor {
  T1 mDeviceStorage;
  T2 mDataAcc;
  KernelFunctor(T1 &DeviceStorage, T2 &DataAcc)
      : mDeviceStorage(DeviceStorage), mDataAcc(DataAcc) {}

  void operator()() const {
    auto *Ptr = mDeviceStorage->template getAs<BaseIncrement>();
    Ptr->increment(
        mDataAcc.template get_multi_ptr<sycl::access::decorated::no>().get());
  }
  auto get(oneapi::properties_tag) const {
    return oneapi::properties{oneapi::assume_indirect_calls};
  }
};

int call(sycl::queue Q, storage_t *DeviceStorage, int Init) {
  int Data = Init;
  {
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(KernelFunctor(DeviceStorage, DataAcc));
    });
  }

  return Data;
}
