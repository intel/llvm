// UNSUPPORTED: cuda, hip, acc
// FIXME: replace unsupported with an aspect check once we have it
//
// RUN: %{build} -o %t.out -Xclang -fsycl-allow-virtual-functions
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <algorithm>

namespace oneapi = sycl::ext::oneapi::experimental;

class Base {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void increment(int *) { /* do nothhing */ }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void multiply(int *) { /* do nothhing */ }

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable<>)
  virtual void substract(int *) { /* do nothhing */ }
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

Base *constructAnObject(char *Storage, int Index) {
  switch (Index) {
  case 0: {
    auto *Ret = reinterpret_cast<IncrementBy1 *>(Storage);
    new (Storage) IncrementBy1;
    return Ret;
  }
  case 1: {
    auto *Ret = reinterpret_cast<IncrementBy1AndSubstractBy2 *>(Storage);
    new (Storage) IncrementBy1AndSubstractBy2;
    return Ret;
  }
  case 2: {
    auto *Ret = reinterpret_cast<MultiplyBy2 *>(Storage);
    new (Storage) MultiplyBy2;
    return Ret;
  }
  case 3: {
    auto *Ret = reinterpret_cast<MultiplyBy2AndIncrementBy8 *>(Storage);
    new (Storage) MultiplyBy2AndIncrementBy8;
    return Ret;
  }
  case 4: {
    auto *Ret = reinterpret_cast<SubstractBy4 *>(Storage);
    new (Storage) SubstractBy4;
    return Ret;
  }
  case 5: {
    auto *Ret = reinterpret_cast<SubstractBy4AndMultiplyBy4 *>(Storage);
    new (Storage) SubstractBy4AndMultiplyBy4;
    return Ret;
  }

  default:
    return nullptr;
  }
}

void applyOp(int *DataPtr, Base *ObjPtr) {
  ObjPtr->increment(DataPtr);
  ObjPtr->substract(DataPtr);
  ObjPtr->multiply(DataPtr);
}

int main() {
  constexpr size_t Size =
      std::max({sizeof(IncrementBy1), sizeof(IncrementBy1AndSubstractBy2),
                sizeof(MultiplyBy2), sizeof(MultiplyBy2AndIncrementBy8),
                sizeof(SubstractBy4), sizeof(SubstractBy4AndMultiplyBy4)});

  sycl::buffer<char> ObjStorage(sycl::range{Size});
  char HostStorage[Size];
  sycl::queue q;

  constexpr oneapi::properties props{oneapi::calls_indirectly<>};
  for (int TestCase = 0; TestCase < 6; ++TestCase) {
    int HostData = 42;
    int Data = HostData;
    sycl::buffer<int> DataStorage(&Data, sycl::range{1});

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor StorageAcc(ObjStorage, CGH, sycl::write_only);
      sycl::accessor DataAcc(DataStorage, CGH, sycl::write_only);
      CGH.single_task(props, [=]() {
        Base *Ptr = constructAnObject(
            StorageAcc.get_multi_ptr<sycl::access::decorated::no>().get(),
            TestCase);
        applyOp(DataAcc.get_multi_ptr<sycl::access::decorated::no>().get(),
                Ptr);
      });
    });

    Base *Ptr = constructAnObject(HostStorage, TestCase);
    applyOp(&HostData, Ptr);

    sycl::host_accessor HostAcc(DataStorage);
    assert(HostAcc[0] == HostData);
  }

  return 0;
}
