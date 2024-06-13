
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/group_sort.hpp>

#pragma once

namespace oneapi_exp = sycl::ext::oneapi::experimental;

enum class UseGroupT { SubGroup = true, WorkGroup = false };

// these classes are needed to pass non-type template parameters to KernelName
template <int> class IntWrapper;
template <UseGroupT> class UseGroupWrapper;

class CustomType {
public:
  CustomType(size_t Val) : MVal(Val) {}
  CustomType() : MVal(0) {}

  bool operator<(const CustomType &RHS) const { return MVal < RHS.MVal; }
  bool operator>(const CustomType &RHS) const { return MVal > RHS.MVal; }
  bool operator==(const CustomType &RHS) const { return MVal == RHS.MVal; }

private:
  size_t MVal = 0;
};

template <class T> struct ConvertToSimpleType {
  using Type = T;
};

// Dummy overloads for CustomType which is not supported by radix sorter
template <> struct ConvertToSimpleType<CustomType> {
  using Type = int;
};

template <class SorterT> struct ConvertToSortingOrder;

template <class T> struct ConvertToSortingOrder<std::greater<T>> {
  static const auto Type = oneapi_exp::sorting_order::descending;
};

template <class T> struct ConvertToSortingOrder<std::less<T>> {
  static const auto Type = oneapi_exp::sorting_order::ascending;
};

constexpr size_t ReqSubGroupSize = 8;

template <typename...> class KernelNameOverGroup;
template <typename...> class KernelNameJoint;
