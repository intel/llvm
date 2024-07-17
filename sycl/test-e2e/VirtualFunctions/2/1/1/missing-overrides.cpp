// UNSUPPORTED: cuda, hip, acc
// FIXME: replace unsupported with an aspect check once we have it
//
// RUN: %{build} -o %t.out -Xclang -fsycl-allow-virtual-functions %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include "helpers.hpp"

#include <iostream>

namespace oneapi = sycl::ext::oneapi::experimental;

class Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void increment(int *) { /* do nothhing */
  }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void multiply(int *) { /* do nothhing */
  }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void substract(int *) { /* do nothhing */
  }
};

class IncrementBy1 : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void increment(int *Data) override { *Data += 1; }
};

class IncrementBy1AndSubstractBy2 : public IncrementBy1 {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void substract(int *Data) override { *Data -= 2; }
};

class MultiplyBy2 : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void multiply(int *Data) override { *Data *= 2; }
};

class MultiplyBy2AndIncrementBy8 : public MultiplyBy2 {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void increment(int *Data) override { *Data += 8; }
};

class SubstractBy4 : public Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void substract(int *Data) override { *Data -= 4; }
};

class SubstractBy4AndMultiplyBy4 : public SubstractBy4 {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  void multiply(int *Data) override { *Data *= 4; }
};

void applyOp(int *DataPtr, Base *ObjPtr) {
  ObjPtr->increment(DataPtr);
  ObjPtr->substract(DataPtr);
  ObjPtr->multiply(DataPtr);
}

int main() try {
  using storage_t = obj_storage_t<IncrementBy1, IncrementBy1AndSubstractBy2,
                                  MultiplyBy2, MultiplyBy2AndIncrementBy8,
                                  SubstractBy4, SubstractBy4AndMultiplyBy4>;
  storage_t HostStorage;
  sycl::buffer<storage_t> DeviceStorage(sycl::range{1});

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);

  constexpr oneapi::properties props{oneapi::calls_indirectly<>};
  for (unsigned TestCase = 0; TestCase < 6; ++TestCase) {
    int HostData = 42;
    int Data = HostData;
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor StorageAcc(DeviceStorage, CGH, sycl::write_only);
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(props, [=]() {
        auto *Ptr = StorageAcc[0].construct</* ret type = */ Base>(TestCase);
        applyOp(DataAcc.get_multi_ptr<sycl::access::decorated::no>().get(),
                Ptr);
      });
    });

    Base *Ptr = HostStorage.construct</* ret type = */ Base>(TestCase);
    applyOp(&HostData, Ptr);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
