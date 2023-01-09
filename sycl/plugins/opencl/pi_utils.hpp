//===--------- pi_utils.hpp - Plugin Utility Functions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#ifndef PI_UTILS_HPP
#define PI_UTILS_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_LEVEL_ZERO_PLUGIN_VERSION 1

#define _PI_LEVEL_ZERO_PLUGIN_VERSION_STRING                                   \
  _PI_PLUGIN_VERSION_STRING(_PI_LEVEL_ZERO_PLUGIN_VERSION)

#define ARG_UNUSED(x) (void)x

namespace {

// Helper functions for unified 'Return' type declaration - imported
// from pi_level_zero.cpp

template <typename T, typename Assign>
pi_result getInfoImpl(size_t ParamValueSize, void *ParamValue,
                      size_t *ParamValueSizeRet, T Value, size_t ValueSize,
                      Assign &&AssignFunc) {
  if (ParamValue != nullptr) {
    if (ParamValueSize < ValueSize) {
      return PI_ERROR_INVALID_VALUE;
    }
    AssignFunc(ParamValue, Value, ValueSize);
  }
  if (ParamValueSizeRet != nullptr) {
    *ParamValueSizeRet = ValueSize;
  }
  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t ParamValueSize, void *ParamValue,
                  size_t *ParamValueSizeRet, T Value) {
  auto assignment = [](void *ParamValue, T Value, size_t ValueSize) {
    ARG_UNUSED(ValueSize);
    *static_cast<T *>(ParamValue) = Value;
  };
  return getInfoImpl(ParamValueSize, ParamValue, ParamValueSizeRet, Value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t ArrayLength, size_t ParamValueSize,
                       void *ParamValue, size_t *ParamValueSizeRet, T *Value) {
  return getInfoImpl(ParamValueSize, ParamValue, ParamValueSizeRet, Value,
                     ArrayLength * sizeof(T), memcpy);
}

class ReturnHelper {
public:
  ReturnHelper(size_t param_value_size, void *param_value,
               size_t *param_value_size_ret)
      : param_value_size(param_value_size), param_value(param_value),
        param_value_size_ret(param_value_size_ret) {}

  template <class T> pi_result operator()(const T &t) {
    return getInfo(param_value_size, param_value, param_value_size_ret, t);
  }

private:
  size_t param_value_size;
  void *param_value;
  size_t *param_value_size_ret;
};

} // anonymous namespace

#endif // PI_UTILS_HPP
