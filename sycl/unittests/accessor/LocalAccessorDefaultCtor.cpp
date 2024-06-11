#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

#include <vector>

using namespace sycl;
using AccT = sycl::local_accessor<int, 1>;

TEST(LocalAccessorDefaultCtorTest, LocalAcessorDefaultCtorIsEmpty) {
  bool empty;

  AccT acc;
  empty = acc.empty();

  EXPECT_TRUE(empty == true);
}

TEST(LocalAccessorDefaultCtorTest, LocalAcessorDefaultCtorSizeQueries) {
  AccT acc;
  size_t byte_size;
  size_t size;
  size_t max_size;

  byte_size = acc.byte_size();
  size = acc.size();
  max_size = acc.max_size();

  EXPECT_TRUE(byte_size == 0);
  EXPECT_TRUE(size == 0);
  EXPECT_TRUE(max_size == 0);
}
