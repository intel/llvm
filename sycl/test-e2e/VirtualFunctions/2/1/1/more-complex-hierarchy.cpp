// RUN: %{build} -o %t.out %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include "helpers.hpp"

#include <iostream>

namespace oneapi = sycl::ext::oneapi::experimental;

class AbstractOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual void applyOp(int *) = 0;
};

class IncrementOp : public AbstractOp {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void applyOp(int *Data) final override { increment(Data); }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual void increment(int *) = 0;
};

class IncrementBy1 : public IncrementOp {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 1; }
};

class IncrementBy2 : public IncrementOp {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 2; }
};

class IncrementBy4 : public IncrementOp {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 4; }
};

class IncrementBy8 : public IncrementOp {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 8; }
};

void applyOp(int *Data, AbstractOp *Obj) { Obj->applyOp(Data); }

int main() try {
  using storage_t =
      obj_storage_t<IncrementBy1, IncrementBy2, IncrementBy4, IncrementBy8>;

  storage_t HostStorage;
  sycl::buffer<storage_t> DeviceStorage(sycl::range{1});

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);

  struct KernelFunctor {
    sycl::buffer<storage_t> mDeviceStorage;
    sycl::buffer<int> mDataStorage;
    sycl::handler mCGH;
    KernelFunctor(sycl::buffer<storage_t> DeviceStorage,
                  sycl::buffer<int> DataStorage, sycl::handler CGH)
        : mDeviceStorage(DeviceStorage), mDataStorage(DataStorage), mCGH(CGH) {}

    void operator()() const {
      sycl::accessor StorageAcc(mDeviceStorage, mCGH, sycl::write_only);
      sycl::accessor DataAcc(mDataStorage, mCGH, sycl::write_only);
      auto *Ptr =
          StorageAcc[0].construct</* ret type = */ AbstractOp>(TestCase);
      applyOp(DataAcc.get_multi_ptr<sycl::access::decorated::no>().get(), Ptr);
    }
    auto get(oneapi::properties_tag) const {
      return oneapi::properties{oneapi::assume_indirect_calls};
    }
  };
  for (unsigned TestCase = 0; TestCase < 4; ++TestCase) {
    int HostData = 42;
    int Data = HostData;
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
      CGH.single_task(KernelFunctor(DeviceStorage, DataStorage, CGH));
    });

    auto *Ptr = HostStorage.construct</* ret type = */ AbstractOp>(TestCase);
    Ptr->applyOp(&HostData);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
