#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using sycl::access::address_space;
using sycl::access::decorated;

template <typename ElementType, address_space Space, decorated IsDecorated>
using multi_ptr_t = sycl::multi_ptr<ElementType, Space, IsDecorated>;

} // namespace

TEST(MultiPtrAccessors, AccessorDevice) {
  using rw_acc = sycl::accessor<int, 1, sycl::access_mode::read_write,
                                sycl::target::device>;
  using atomic_acc =
      sycl::accessor<int, 1, sycl::access_mode::atomic, sycl::target::device>;
  using ro_acc =
      sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::device>;
  using global_multi_ptr =
      sycl::multi_ptr<int, sycl::access::address_space::global_space>;
  using generic_multi_ptr =
      sycl::multi_ptr<int, sycl::access::address_space::generic_space>;
  using const_multi_ptr =
      sycl::multi_ptr<const int, sycl::access::address_space::generic_space>;

  using float_global_ptr =
      sycl::multi_ptr<float, sycl::access::address_space::global_space>;

  EXPECT_TRUE((std::is_constructible_v<global_multi_ptr, rw_acc>));
  EXPECT_TRUE((std::is_constructible_v<generic_multi_ptr, rw_acc>));
  EXPECT_TRUE((std::is_constructible_v<global_multi_ptr, atomic_acc>));
  EXPECT_TRUE((std::is_constructible_v<global_multi_ptr, ro_acc>));
  EXPECT_TRUE((std::is_constructible_v<generic_multi_ptr, ro_acc>));
  EXPECT_FALSE((std::is_constructible_v<float_global_ptr, rw_acc>));
  EXPECT_FALSE((std::is_constructible_v<float_global_ptr, ro_acc>));
  EXPECT_TRUE((std::is_constructible_v<const_multi_ptr, rw_acc>));
  EXPECT_TRUE((std::is_constructible_v<const_multi_ptr, ro_acc>));
}

TEST(MultiPtrAccessors, LocalAccessor) {
  using local_acc = sycl::local_accessor<int, 1>;
  using local_multi_ptr =
      sycl::multi_ptr<int, sycl::access::address_space::local_space>;
  using generic_multi_ptr =
      sycl::multi_ptr<int, sycl::access::address_space::generic_space>;
  using const_multi_ptr =
      sycl::multi_ptr<const int, sycl::access::address_space::generic_space>;
  using float_local_ptr =
      sycl::multi_ptr<float, sycl::access::address_space::local_space>;

  EXPECT_TRUE((std::is_constructible_v<local_multi_ptr, local_acc>));
  EXPECT_TRUE((std::is_constructible_v<generic_multi_ptr, local_acc>));
  EXPECT_FALSE((std::is_constructible_v<
                sycl::multi_ptr<int, sycl::access::address_space::global_space>,
                local_acc>));
  EXPECT_FALSE((std::is_constructible_v<float_local_ptr, local_acc>));
  EXPECT_TRUE((std::is_constructible_v<const_multi_ptr, local_acc>));
}

TEST(MultiPtrAccessors, AddrSpaceCast) {
  int Value = 42;
  const int ConstValue = 17;

  auto PrivateUndecorated =
      sycl::address_space_cast<address_space::private_space, decorated::no>(
          &Value);
  auto GlobalDecorated =
      sycl::address_space_cast<address_space::global_space, decorated::yes>(
          &Value);
  auto LocalUndecorated =
      sycl::address_space_cast<address_space::local_space, decorated::no>(
          &Value);
  auto GenericDecorated =
      sycl::address_space_cast<address_space::generic_space, decorated::yes>(
          &Value);
  auto ConstPrivateDecorated =
      sycl::address_space_cast<address_space::private_space, decorated::yes>(
          &ConstValue);

  EXPECT_TRUE((std::is_same_v<
               decltype(PrivateUndecorated),
               multi_ptr_t<int, address_space::private_space, decorated::no>>));
  EXPECT_TRUE((std::is_same_v<
               decltype(GlobalDecorated),
               multi_ptr_t<int, address_space::global_space, decorated::yes>>));
  EXPECT_TRUE((std::is_same_v<
               decltype(LocalUndecorated),
               multi_ptr_t<int, address_space::local_space, decorated::no>>));
  EXPECT_TRUE(
      (std::is_same_v<
          decltype(GenericDecorated),
          multi_ptr_t<int, address_space::generic_space, decorated::yes>>));
  EXPECT_TRUE(
      (std::is_same_v<decltype(ConstPrivateDecorated),
                      multi_ptr_t<const int, address_space::private_space,
                                  decorated::yes>>));

  EXPECT_EQ(PrivateUndecorated.get_raw(), &Value);
  EXPECT_EQ(GlobalDecorated.get_raw(), &Value);
  EXPECT_EQ(LocalUndecorated.get_raw(), &Value);
  EXPECT_EQ(GenericDecorated.get_raw(), &Value);
  EXPECT_EQ(ConstPrivateDecorated.get_raw(), &ConstValue);

  EXPECT_FALSE((std::is_same_v<
                decltype(PrivateUndecorated),
                multi_ptr_t<int, address_space::global_space, decorated::no>>));
  EXPECT_FALSE(
      (std::is_same_v<
          decltype(PrivateUndecorated),
          multi_ptr_t<int, address_space::private_space, decorated::yes>>));
  EXPECT_FALSE(
      (std::is_same_v<
          decltype(ConstPrivateDecorated),
          multi_ptr_t<int, address_space::private_space, decorated::yes>>));

  auto NullPrivate =
      sycl::address_space_cast<address_space::private_space, decorated::no>(
          static_cast<int *>(nullptr));
  auto NullGlobalConst =
      sycl::address_space_cast<address_space::global_space, decorated::yes>(
          static_cast<const int *>(nullptr));

  EXPECT_EQ(NullPrivate, nullptr);
  EXPECT_EQ(NullGlobalConst, nullptr);
}
