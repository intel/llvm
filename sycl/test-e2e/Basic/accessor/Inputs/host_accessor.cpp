//==----------------host_accessor.cpp - SYCL accessor basic test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <sycl/detail/core.hpp>

int main() {
  {
    int data[3] = {3, 7, 9};

    sycl::buffer<int, 1> buf_data(data, sycl::range<1>(3),
                                  {sycl::property::buffer::use_host_ptr()});

    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access();
#endif

      assert(acc.size() == 3);
      assert(acc.get_range() == sycl::range<1>(3));
      assert(acc[0] == 3);

      acc[0] = 2;

      assert(data[0] == 2 && data[1] == 7 && data[2] == 9);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::read_only);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::read_only);
#endif

      assert(acc.size() == 3);
      assert(acc.get_range() == sycl::range<1>(3));
      assert(acc[0] == 2);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::write_only);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::write_only);
#endif

      assert(acc.size() == 3);
      assert(acc.get_range() == sycl::range<1>(3));
      acc[0] = 1;
      assert(data[0] == 1 && data[1] == 7 && data[2] == 9);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::range<1>(2));
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::range<1>(2));
#endif

      assert(acc.size() == 2);
      assert(acc.get_range() == sycl::range<1>(2));
      assert(acc[0] == 1);

      acc[0] = 2;

      assert(data[0] == 2 && data[1] == 7 && data[2] == 9);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::range<1>(2), sycl::read_only);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::range<1>(2), sycl::read_only);
#endif

      assert(acc.size() == 2);
      assert(acc.get_range() == sycl::range<1>(2));
      assert(acc[0] == 2);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::range<1>(2), sycl::write_only);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::range<1>(2), sycl::write_only);
#endif

      assert(acc.size() == 2);
      assert(acc.get_range() == sycl::range<1>(2));
      acc[0] = 1;
      assert(data[0] == 1 && data[1] == 7 && data[2] == 9);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::range<1>(2), sycl::id<1>(1));
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::range<1>(2), sycl::id<1>(1));
#endif

      assert(acc.size() == 2);
      assert(acc.get_range() == sycl::range<1>(2));
      assert(acc[0] == 7);

      acc[0] = 6;

      assert(data[0] == 1 && data[1] == 6 && data[2] == 9);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::range<1>(2), sycl::id<1>(1),
                              sycl::read_only);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::range<1>(2), sycl::id<1>(1),
                                          sycl::read_only);
#endif

      assert(acc.size() == 2);
      assert(acc.get_range() == sycl::range<1>(2));
      assert(acc[0] == 6);
    }
    {
#if defined(accessor_new_api_test)
      sycl::host_accessor acc(buf_data, sycl::range<1>(2), sycl::id<1>(1),
                              sycl::write_only);
#elif defined(buffer_new_api_test)
      auto acc = buf_data.get_host_access(sycl::range<1>(2), sycl::id<1>(1),
                                          sycl::write_only);
#endif

      assert(acc.size() == 2);
      assert(acc.get_range() == sycl::range<1>(2));
      acc[0] = 5;
      assert(data[0] == 1 && data[1] == 5 && data[2] == 9);
    }
  }
}
