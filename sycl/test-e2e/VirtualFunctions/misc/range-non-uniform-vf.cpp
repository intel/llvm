// REQUIRES: aspect-usm_shared_allocations
//
// This test checks that virtual functions work correctly within simple range
// kernels when different work-items perform a virtual function calls using
// different objects.
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
  T1 mDeviceStorage;
  T2 mDataAcc;
  KernelFunctor(T1 DeviceStorage, T2 DataAcc)
      : mDeviceStorage(DeviceStorage), mDataAcc(DataAcc) {}

  void operator()(sycl::id<1> It) const {
    // Select an object that corresponds to this work-item
    auto Ind = It % 3;
    auto *Ptr = mDeviceStorage[Ind].template getAs<BaseOp>();
    mDataAcc[It] = Ptr->apply(mDataAcc[It]);
  }
  auto get(oneapi::properties_tag) const {
    return oneapi::properties{oneapi::assume_indirect_calls};
  }
};

int main() try {
  using storage_t = obj_storage_t<FloorOp, CeilOp, RoundOp>;

  std::array<storage_t, 3> HostStorage;

  sycl::queue q;

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(3, q);
  sycl::range R{1024};

  {
    std::vector<float> HostData(R.size());
    for (size_t I = 1; I < HostData.size(); ++I)
      HostData[I] = HostData[I - 1] + 0.7;
    std::vector<float> DeviceData = HostData;
    sycl::buffer<float> DataStorage(DeviceData.data(), R);

    q.submit([&](sycl::handler &CGH) {
       CGH.single_task([=]() {
         DeviceStorage[0].construct</* ret type = */ BaseOp>(0);
         DeviceStorage[1].construct</* ret type = */ BaseOp>(1);
         DeviceStorage[2].construct</* ret type = */ BaseOp>(2);
       });
     }).wait_and_throw();

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::read_write);
      CGH.parallel_for(R, KernelFunctor(DeviceStorage, DataAcc));
    });

    BaseOp *Ptr[] = {HostStorage[0].construct</* ret type = */ BaseOp>(0),
                     HostStorage[1].construct</* ret type = */ BaseOp>(1),
                     HostStorage[2].construct</* ret type = */ BaseOp>(2)};

    for (size_t I = 0; I < HostData.size(); ++I)
      HostData[I] = Ptr[I % 3]->apply(HostData[I]);

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
