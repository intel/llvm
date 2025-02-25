#pragma once

#include <cassert>
#include <iostream>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/marray.hpp>
#include <sycl/usm.hpp>
#include <sycl/vector.hpp>

using namespace sycl;

template <typename T> bool check_half_aspect(queue &q) {
  if (std::is_same_v<sycl::half, T> &&
      !q.get_device().has(sycl::aspect::fp16)) {
    std::cout << "Device does not support fp16 aspect. Skipping all tests with "
                 "sycl::half type!"
              << std::endl;
    return false;
  }
  return true;
}

template <typename T> bool check_double_aspect(queue &q) {
  if (std::is_same_v<T, double> && !q.get_device().has(aspect::fp64)) {
    std::cout << "Device does not support fp64 aspect. Skipping all tests with "
                 "double type!"
              << std::endl;
    return false;
  }
  return true;
}

template <typename T> struct S {
  T val;
};

template <typename T> struct M {
  T val;
};

union U {
  S<int> s;
  M<int> m;
};

template <typename T>
void sum_helper(sycl::ext::oneapi::experimental::work_group_memory<T[]> mem,
                sycl::ext::oneapi::experimental::work_group_memory<T> ret,
                size_t WGSIZE) {
  for (int i = 0; i < WGSIZE; ++i) {
    ret = ret + mem[i];
  }
}
