#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

TEST(AccessorImplicitConversionTest, NonConstToConstTypeInReadAccessor) {
  using accessor_non_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read, sycl::target::device>;
  using acc_const_type = sycl::accessor<const int, 1, sycl::access::mode::read,
                                        sycl::target::device>;
  accessor_non_const_type acc_a;
  auto acc_b = acc_const_type(acc_a);
  acc_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a, acc_b);
  EXPECT_EQ(acc_a, acc_c);
}

TEST(AccessorImplicitConversionTest, ConstToNonConstTypeInReadAccessor) {
  using accessor_non_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read, sycl::target::device>;
  using acc_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read, sycl::target::device>;
  acc_const_type acc_a;
  auto acc_b = accessor_non_const_type(acc_a);
  accessor_non_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a, acc_b);
  EXPECT_EQ(acc_a, acc_c);
}

TEST(AccessorImplicitConversionTest, ReadWriteAccessorToReadAccessorNonConst) {
  using read_write_accessor_non_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::target::device>;
  using read_accessor_non_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read, sycl::target::device>;
  read_write_accessor_non_const_type acc_a;
  auto acc_b = read_accessor_non_const_type(acc_a);
  read_accessor_non_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a.impl, acc_b.impl);
  EXPECT_EQ(acc_a.impl, acc_c.impl);
}

TEST(AccessorImplicitConversionTest, ReadWriteAccessorToReadAccessorConst) {
  using read_write_accessor_non_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::target::device>;
  using read_accessor_non_const_type =
      sycl::accessor<const int, 1, sycl::access::mode::read,
                     sycl::target::device>;
  read_write_accessor_non_const_type acc_a;
  auto acc_b = read_accessor_non_const_type(acc_a);
  read_accessor_non_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a.impl, acc_b.impl);
  EXPECT_EQ(acc_a.impl, acc_c.impl);
}
