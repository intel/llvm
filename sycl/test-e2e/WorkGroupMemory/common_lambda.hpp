#pragma once

#include <cassert>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/marray.hpp>
#include <sycl/usm.hpp>
#include <sycl/vector.hpp>

using namespace sycl;

template <typename T>
void sum_helper(sycl::ext::oneapi::experimental::work_group_memory<T[]> mem,
                sycl::ext::oneapi::experimental::work_group_memory<T> ret,
                size_t WGSIZE) {
  for (int i = 0; i < WGSIZE; ++i) {
    ret = ret + mem[i];
  }
}
