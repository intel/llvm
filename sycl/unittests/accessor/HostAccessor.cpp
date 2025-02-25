#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

TEST(HostAccessor, ZeroDimAccessor) {
  using DataT = int;
  using AccT = sycl::host_accessor<DataT, 0>;

  DataT data = 42;
  sycl::buffer<DataT, 1> data_buf(&data, sycl::range<1>(1));
  AccT acc = {data_buf};

  ASSERT_EQ(acc.get_size(), sizeof(DataT));
  ASSERT_EQ(acc.size(), static_cast<size_t>(1));

  DataT &ref = acc;
  ASSERT_EQ(ref, data);
}
