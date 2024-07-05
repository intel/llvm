
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

template <typename Properties>
constexpr bool IsOutputBlocked(Properties properties) {
  if constexpr (properties.template has_property<
                    oneapi_exp::output_data_placement_key>())
    return properties.template get_property<
               oneapi_exp::output_data_placement_key>() ==
           oneapi_exp::output_data_placement<
               oneapi_exp::group_algorithm_data_placement::blocked>;
  else
    return true;
}

// Write data from In to Out in blocked/striped way.
template <typename T, typename Properties>
void writeBlockedOrStriped(const std::vector<T> &In, std::vector<T> &Out,
                           size_t MaxGroupSize, size_t ElementsPerWorkItem,
                           Properties Prop) {
  assert(In.size() == Out.size());
  size_t index = {};
  size_t shift = {};
  auto ChunkSize = MaxGroupSize * ElementsPerWorkItem;
  std::uint32_t ChunkStart = 0;
  for (std::uint32_t ChunkStart = 0; ChunkStart < In.size();
       ChunkStart += ChunkSize) {
    auto GroupSize = (In.size() - ChunkStart) >= ChunkSize
                         ? MaxGroupSize
                         : (In.size() - ChunkStart) / ElementsPerWorkItem;
    for (std::uint32_t j = 0; j < GroupSize; ++j) {
      for (std::uint32_t k = 0; k < ElementsPerWorkItem; ++k) {
        index = ChunkStart + j * ElementsPerWorkItem + k;
        if constexpr (IsOutputBlocked(Prop)) {
          shift = index;
        } else {
          shift = ChunkStart + k * GroupSize + j;
        }
        if (index < Out.size() && shift < In.size())
          Out[index] = In[shift];
      }
    }
  }
}
