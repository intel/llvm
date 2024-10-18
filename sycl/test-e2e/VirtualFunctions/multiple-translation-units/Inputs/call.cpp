#include "declarations.hpp"

int call(sycl::queue Q, storage_t *DeviceStorage, int Init) {
  int Data = Init;
  {
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});
    constexpr oneapi::properties props{oneapi::assume_indirect_calls};
    Q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(props, [=]() {
        auto *Ptr = DeviceStorage->getAs<BaseIncrement>();
        Ptr->increment(
            DataAcc.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });
  }

  return Data;
}
