#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using AccT = sycl::host_accessor<int, 0>;

TEST(HostAccessorAssignmentOpTest, AreEqual) {
  int Data = 0;
  sycl::buffer<int, 1> DataBuffer(&Data, sycl::range<1>(1));
  AccT HostAccessor(DataBuffer);

  HostAccessor = 1;

  AccT::reference &HostAccRef = HostAccessor;
  EXPECT_TRUE(HostAccRef == 1);
}
