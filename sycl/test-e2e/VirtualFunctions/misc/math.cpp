// REQUIRES: aspect-usm_shared_allocations
//
// This test checks that SYCL math built-in functions work correctly
// inside virtual functions.
//
// RUN: %{build} -o %t.out %helper-includes
// RUN: %{run} %t.out

#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

#include <iostream>

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual float apply(float) = 0;
};

class FloorOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual float apply(float V) { return sycl::floor(V); }
};

class CeilOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual float apply(float V) { return sycl::ceil(V); }
};

class RoundOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual float apply(float V) { return sycl::round(V); }
};

int main() try {
  using storage_t = obj_storage_t<FloorOp, CeilOp, RoundOp>;

  storage_t HostStorage;

  sycl::queue q;

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);

  template <typename T> struct KernelFunctor {
    T mDeviceStorage;
    sycl::buffer<int> mDataStorage;
    sycl::handler mCGH;
    KernelFunctor(T DeviceStorage, sycl::buffer<float> DataStorage,
                  sycl::handler CGH) {
      mDeviceStorage = DeviceStorage;
      mDataStorage = DataStorage;
      mCGH = CGH;
    }

    void operator()() const {
      sycl::accessor DataAcc(mDataStorage, mCGH, sycl::read_write);
      auto *Ptr = DeviceStorage->getAs<BaseOp>();
      DataAcc[0] = Ptr->apply(DataAcc[0]);
    }
    auto get(oneapi::properties_tag) const {
      return oneapi::properties{oneapi::assume_indirect_calls};
    }
  };
  for (unsigned TestCase = 0; TestCase < 3; ++TestCase) {
    float HostData = 3.56;
    float Data = HostData;
    sycl::buffer<float> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
       CGH.single_task([=]() {
         DeviceStorage->construct</* ret type = */ BaseOp>(TestCase);
       });
     }).wait_and_throw();

    q.submit([&](sycl::handler &CGH) {
      CGH.single_task(KernelFunctor(DeviceStorage, DataStorage, CGH));
    });

    auto *Ptr = HostStorage.construct</* ret type = */ BaseOp>(TestCase);
    HostData = Ptr->apply(HostData);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  sycl::free(DeviceStorage, q);

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
