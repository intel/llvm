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

TEST(LocalAccessorDefaultCtorTest, LocalAcessorDefaultCtorPtrQueries) {
  AccT acc;

  // The return values of get_pointer() and get_multi_ptr() are
  // unspecified. Just check they can run without any issue.
  auto ptr = acc.get_pointer();
  (void)ptr;
  // TODO: uncomment check with get_multi_ptr() when SYCL 2020 multi_ptr feature
  // will be merged
  // auto multi_ptr = acc.get_multi_ptr();
  // (void)multi_ptr;
}
