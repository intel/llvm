// REQUIRES: aspect-usm_shared_allocations
//
// This test checks that virtual functions work correctly within simple range
// kernels when every work-item calls the same virtual function on the same
// object.
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
  sycl::range R{1024};

  constexpr oneapi::properties props{oneapi::assume_indirect_calls};
  for (unsigned TestCase = 0; TestCase < 3; ++TestCase) {
    std::vector<float> HostData(R.size());
    for (size_t I = 1; I < HostData.size(); ++I)
      HostData[I] = HostData[I - 1] + 0.7;
    std::vector<float> DeviceData = HostData;
    sycl::buffer<float> DataStorage(DeviceData.data(), R);

    q.submit([&](sycl::handler &CGH) {
       CGH.single_task([=]() {
         DeviceStorage->construct</* ret type = */ BaseOp>(TestCase);
       });
     }).wait_and_throw();

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::read_write);
      CGH.parallel_for(R, props, [=](auto it) {
        auto *Ptr = DeviceStorage->getAs<BaseOp>();
        DataAcc[it] = Ptr->apply(DataAcc[it]);
      });
    });

    auto *Ptr = HostStorage.construct</* ret type = */ BaseOp>(TestCase);
    for (size_t I = 0; I < HostData.size(); ++I)
      HostData[I] = Ptr->apply(HostData[I]);

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
