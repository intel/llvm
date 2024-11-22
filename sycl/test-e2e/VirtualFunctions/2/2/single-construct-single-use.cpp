// RUN: %{build} -o %t.out %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include "helpers.hpp"

#include <iostream>

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseIncrement {
public:
  BaseIncrement(int Mod, int /* unused */ = 42) : Mod(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual void increment(int *Data) { *Data += 1 + Mod; }

protected:
  int Mod = 0;
};

class IncrementBy2 : public BaseIncrement {
public:
  IncrementBy2(int Mod, int /* unused */) : BaseIncrement(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 2 + Mod; }
};

class IncrementBy4 : public BaseIncrement {
public:
  IncrementBy4(int Mod, int ExtraMod)
      : BaseIncrement(Mod), ExtraMod(ExtraMod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 4 + Mod + ExtraMod; }

private:
  int ExtraMod = 0;
};

class IncrementBy8 : public BaseIncrement {
public:
  IncrementBy8(int Mod, int /* unused */) : BaseIncrement(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override { *Data += 8 + Mod; }
};

struct SetIncBy16;
class IncrementBy16 : public BaseIncrement {
public:
  IncrementBy16(int Mod, int /* unused */) : BaseIncrement(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable_in<SetIncBy16>)
  void increment(int *Data) override { *Data += 16 + Mod; }
};

int main() try {
  using storage_t = obj_storage_t<BaseIncrement, IncrementBy2, IncrementBy4,
                                  IncrementBy8, IncrementBy16>;

  storage_t HostStorage;
  sycl::buffer<storage_t> DeviceStorage(sycl::range{1});

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);

  // TODO: cover uses case when objects are passed through USM
  constexpr oneapi::properties props{
      oneapi::assume_indirect_calls_to<void, SetIncBy16>};
  for (unsigned TestCase = 0; TestCase < 5; ++TestCase) {
    int HostData = 42;
    int Data = HostData;
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor StorageAcc(DeviceStorage, CGH, sycl::write_only);
      CGH.single_task([=]() {
        StorageAcc[0].construct</* ret type = */ BaseIncrement>(TestCase, 19,
                                                                23);
      });
    });

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor StorageAcc(DeviceStorage, CGH, sycl::read_write);
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(props, [=]() {
        auto *Ptr = StorageAcc[0].getAs<BaseIncrement>();
        Ptr->increment(
            DataAcc.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    auto *Ptr =
        HostStorage.construct</* ret type = */ BaseIncrement>(TestCase, 19, 23);
    Ptr->increment(&HostData);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
