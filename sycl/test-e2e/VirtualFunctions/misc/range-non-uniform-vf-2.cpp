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

int main() try {
  using storage_t = obj_storage_t<OpA, OpB>;

  storage_t HostStorage;

  sycl::queue q;

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);
  sycl::range R{1024};

  constexpr oneapi::properties props{oneapi::assume_indirect_calls};
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
      CGH.parallel_for(R, props, [=](auto It) {
        // Select method that corresponds to this work-item
        auto *Ptr = DeviceStorage->template getAs<BaseOp>();
        if (It % 2)
          DataAcc[It] = Ptr->foo(DataAcc[It]);
        else
          DataAcc[It] = Ptr->bar(DataAcc[It]);
      });
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
