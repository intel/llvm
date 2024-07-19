// UNSUPPORTED: cuda, hip, acc
// FIXME: replace unsupported with an aspect check once we have it
//
// RUN: %{build} -o %t.out -Xclang -fsycl-allow-virtual-functions %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include "helpers.hpp"

#include <iostream>

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseIncrement {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void increment(int *Data) { *Data += 1; }
};

class IncrementBy2 : public BaseIncrement {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void increment(int *Data) override { *Data += 2; }
};

class IncrementBy4 : public BaseIncrement {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void increment(int *Data) override { *Data += 4; }
};

class IncrementBy8 : public BaseIncrement {
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void increment(int *Data) override { *Data += 8; }
};

int main() try {
  using storage_t =
      obj_storage_t<BaseIncrement, IncrementBy2, IncrementBy4, IncrementBy8>;

  storage_t HostStorage;
  sycl::buffer<storage_t> DeviceStorage(sycl::range{1});

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);

  constexpr oneapi::properties props{oneapi::calls_indirectly<>};
  for (unsigned TestCase = 0; TestCase < 4; ++TestCase) {
    int HostData = 42;
    int Data = HostData;
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor StorageAcc(DeviceStorage, CGH, sycl::write_only);
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(props, [=]() {
        auto *Ptr =
            StorageAcc[0].construct</* ret type = */ BaseIncrement>(TestCase);
        Ptr->increment(
            DataAcc.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    auto *Ptr = HostStorage.construct</* ret type = */ BaseIncrement>(TestCase);
    Ptr->increment(&HostData);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
