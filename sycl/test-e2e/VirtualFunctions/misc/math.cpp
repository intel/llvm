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

template <typename T1, typename T2> struct KernelFunctor {
  T1 mDataAcc;
  T2 mDeviceStorage;
  KernelFunctor(T1 &DataAcc, T2 &DeviceStorage)
      : mDataAcc(DataAcc), mDeviceStorage(DeviceStorage) {}

  void operator()() const {
    auto *Ptr = mDeviceStorage->getAs<BaseOp>();
    mDataAcc[0] = Ptr->apply(mDataAcc[0]);
  }
  auto get(oneapi::properties_tag) const {
    return oneapi::properties{oneapi::assume_indirect_calls};
  }
};

int main() try {
  using storage_t = obj_storage_t<FloorOp, CeilOp, RoundOp>;

  storage_t HostStorage;

  sycl::queue q;

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);

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
      sycl::accessor DataAcc(DataStorage, CGH, sycl::read_write);
      CGH.single_task(KernelFunctor(DataAcc, DeviceStorage));
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
