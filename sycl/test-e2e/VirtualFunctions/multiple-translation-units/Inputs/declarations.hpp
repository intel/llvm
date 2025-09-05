#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseIncrement {
public:
  BaseIncrement(int Mod, int /* unused */ = 42) : Mod(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual void increment(int *Data);

protected:
  int Mod = 0;
};

class IncrementBy2 : public BaseIncrement {
public:
  IncrementBy2(int Mod, int /* unused */) : BaseIncrement(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override;
};

class IncrementBy4 : public BaseIncrement {
public:
  IncrementBy4(int Mod, int ExtraMod)
      : BaseIncrement(Mod), ExtraMod(ExtraMod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override;

private:
  int ExtraMod = 0;
};

class IncrementBy8 : public BaseIncrement {
public:
  IncrementBy8(int Mod, int /* unused */) : BaseIncrement(Mod) {}

  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  void increment(int *Data) override;
};

using storage_t =
    obj_storage_t<BaseIncrement, IncrementBy2, IncrementBy4, IncrementBy8>;

void construct(sycl::queue Q, storage_t *DeviceStorage, unsigned TestCase);
int call(sycl::queue Q, storage_t *DeviceStorage, int Init);
