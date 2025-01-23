// REQUIRES: aspect-usm_shared_allocations
//
// This test checks that virtual functions work correctly in simple range
// kernels when different work-items perform calls to different virtual
// functions using the same object.
//
// RUN: %{build} -o %t.out %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

#include <iostream>
#include <numeric>

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int foo(int) = 0;

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int bar(int) = 0;
};

class OpA : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int foo(int V) { return V + 2; }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int bar(int V) { return V - 2; }
};

class OpB : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int foo(int V) { return V * 2; }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int bar(int V) { return V / 2; }
};

template <typename T1, typename T2> struct KernelFunctor {
  T1 mDeviceStorage;
  T2 mDataAcc;
  KernelFunctor(T1 DeviceStorage, T2 DataAcc)
      : mDeviceStorage(DeviceStorage), mDataAcc(DataAcc) {}

  void operator()(sycl::id<1> It) const {
    // Select method that corresponds to this work-item
    auto *Ptr = mDeviceStorage->template getAs<BaseOp>();
    if (It % 2)
      mDataAcc[It] = Ptr->foo(mDataAcc[It]);
    else
      mDataAcc[It] = Ptr->bar(mDataAcc[It]);
  }
  auto get(oneapi::properties_tag) const {
    return oneapi::properties{oneapi::assume_indirect_calls};
  }
};

int main() try {
  using storage_t = obj_storage_t<OpA, OpB>;

  storage_t HostStorage;

  sycl::queue q;

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);
  sycl::range R{1024};

  for (size_t TestCase = 0; TestCase < 2; ++TestCase) {
    std::vector<int> HostData(R.size());
    std::iota(HostData.begin(), HostData.end(), 0);
    std::vector<int> DeviceData = HostData;
    sycl::buffer<int> DataStorage(DeviceData.data(), R);

    q.submit([&](sycl::handler &CGH) {
       CGH.single_task([=]() {
         DeviceStorage->construct</* ret type = */ BaseOp>(TestCase);
       });
     }).wait_and_throw();

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::read_write);
      CGH.parallel_for(R, KernelFunctor(DeviceStorage, DataAcc));
    });

    BaseOp *Ptr = HostStorage.construct</* ret type = */ BaseOp>(TestCase);

    for (size_t I = 0; I < HostData.size(); ++I) {
      if (I % 2)
        HostData[I] = Ptr->foo(HostData[I]);
      else
        HostData[I] = Ptr->bar(HostData[I]);
    }

    sycl::host_accessor HostAcc(DataStorage);
    for (size_t I = 0; I < HostData.size(); ++I)
      assert(HostAcc[I] == HostData[I]);
  }

  sycl::free(DeviceStorage, q);

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
