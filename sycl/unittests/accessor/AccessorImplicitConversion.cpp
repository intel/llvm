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

TEST(AccessorImplicitConversionTest, NonConstToConstTypeInReadHostAccessor) {
  using host_accessor_non_const_type =
      sycl::accessor<int, 1, sycl::access::mode::read>;
  using host_accessor_const_type =
      sycl::accessor<const int, 1, sycl::access::mode::read>;
  host_accessor_non_const_type acc_a;
  auto acc_b = host_accessor_const_type(acc_a);
  host_accessor_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a, acc_b);
  EXPECT_EQ(acc_a, acc_c);
}

TEST(AccessorImplicitConversionTest, ConstToNonConstTypeInReadHostAccessor) {
  using accessor_non_const_type =
      sycl::host_accessor<const int, 1, sycl::access::mode::read>;
  using acc_const_type = sycl::host_accessor<int, 1, sycl::access::mode::read>;
  acc_const_type acc_a;
  auto acc_b = accessor_non_const_type(acc_a);
  accessor_non_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a, acc_b);
  EXPECT_EQ(acc_a, acc_c);
}

TEST(AccessorImplicitConversionTest,
     ReadWriteHostAccessorToReadHostAccessorNonConst) {
  using read_write_host_accessor_non_const_type =
      sycl::host_accessor<int, 1, sycl::access::mode::read_write>;
  using read_host_accessor_non_const_type =
      sycl::host_accessor<int, 1, sycl::access::mode::read>;
  read_write_host_accessor_non_const_type acc_a;
  auto acc_b = read_host_accessor_non_const_type(acc_a);
  read_host_accessor_non_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a.impl, acc_b.impl);
  EXPECT_EQ(acc_a.impl, acc_c.impl);
}

TEST(AccessorImplicitConversionTest,
     ReadWriteHostAccessorToReadHostAccessorConst) {
  using read_write_host_accessor_non_const_type =
      sycl::host_accessor<int, 1, sycl::access::mode::read_write>;
  using read_host_accessor_non_const_type =
      sycl::host_accessor<const int, 1, sycl::access::mode::read>;
  read_write_host_accessor_non_const_type acc_a;
  auto acc_b = read_host_accessor_non_const_type(acc_a);
  read_host_accessor_non_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a.impl, acc_b.impl);
  EXPECT_EQ(acc_a.impl, acc_c.impl);
}

TEST(AccessorImplicitConversionTest, NonConstToConstTypeInReadLocalAccessor) {
  using local_accessor_non_const_type = sycl::local_accessor<int, 1>;
  using local_accessor_const_type = sycl::local_accessor<const int, 1>;
  local_accessor_non_const_type acc_a;
  auto acc_b = local_accessor_const_type(acc_a);
  local_accessor_const_type acc_c = acc_a;
  EXPECT_EQ(acc_a, acc_b);
  EXPECT_EQ(acc_a, acc_c);
}
