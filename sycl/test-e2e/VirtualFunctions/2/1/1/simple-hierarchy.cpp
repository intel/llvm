// UNSUPPORTED: cuda, hip, acc
// FIXME: replace unsupported with an aspect check once we have it
//
// RUN: %{build} -o %t.out -Xclang -fsycl-allow-virtual-functions
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

#include <algorithm>

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

BaseIncrement *constructAnObject(char *Storage, int Index) {
  switch (Index) {
  case 0: {
    auto *Ret = reinterpret_cast<BaseIncrement *>(Storage);
    new (Storage) BaseIncrement;
    return Ret;
  }
  case 1: {
    auto *Ret = reinterpret_cast<IncrementBy2 *>(Storage);
    new (Storage) IncrementBy2;
    return Ret;
  }
  case 2: {
    auto *Ret = reinterpret_cast<IncrementBy4 *>(Storage);
    new (Storage) IncrementBy4;
    return Ret;
  }
  case 3: {
    auto *Ret = reinterpret_cast<IncrementBy8 *>(Storage);
    new (Storage) IncrementBy8;
    return Ret;
  }

  default:
    return nullptr;
  }
}

int main() {
  constexpr size_t Size =
      std::max({sizeof(BaseIncrement), sizeof(IncrementBy2),
                sizeof(IncrementBy4), sizeof(IncrementBy8)});

  sycl::buffer<char> ObjStorage(sycl::range{Size});
  char HostStorage[Size];
  sycl::queue q;

  constexpr oneapi::properties props{oneapi::calls_indirectly<>};
  for (int TestCase = 0; TestCase < 4; ++TestCase) {
    int HostData = 42;
    int Data = HostData;
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor StorageAcc(ObjStorage, CGH, sycl::write_only);
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(props, [=]() {
        BaseIncrement *Ptr = constructAnObject(
            StorageAcc.get_multi_ptr<sycl::access::decorated::no>().get(),
            TestCase);
        Ptr->increment(
            DataAcc.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    BaseIncrement *Ptr = constructAnObject(HostStorage, TestCase);
    Ptr->increment(&HostData);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  return 0;
}
