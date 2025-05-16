#pragma once

#ifdef __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <sycl/ext/oneapi/free_function_queries.hpp>

#define SYCL_KHR_WORK_ITEM_QUERIES 1

namespace sycl {
inline namespace _V1 {
namespace khr {

template <int Dimensions> nd_item<Dimensions> this_nd_item() {
  return ext::oneapi::experimental::this_nd_item<Dimensions>();
}

template <int Dimensions> group<Dimensions> this_group() {
  return ext::oneapi::this_work_item::get_work_group<Dimensions>();
}

inline sub_group this_sub_group() {
  return ext::oneapi::this_work_item::get_sub_group();
}

} // namespace khr
} // namespace _V1
} // namespace sycl

#endif
