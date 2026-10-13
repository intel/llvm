//==-- KhrUSMProperties.cpp --- sycl_khr_properties USM allocations --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <gtest/gtest.h>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <map>
#include <type_traits>

namespace kp = sycl::khr::property;
using namespace sycl::khr;
using sycl::usm::alloc;

// Test-only property that is valid for host allocations only.
namespace test_props {
namespace key {
struct host_only : kp::detail::runtime_property_key {};
} // namespace key
struct host_only : kp::detail::runtime_property<key::host_only> {
  int value = 0;
};
} // namespace test_props

namespace sycl {
inline namespace _V1 {
namespace khr {
template <>
struct is_property_key_for<test_props::key::host_only,
                           khr::property::tag::usm_host> : std::true_type {};
} // namespace khr
} // namespace _V1
} // namespace sycl

static_assert(is_property_list_for_v<empty_properties_t, kp::tag::usm_device>);
static_assert(is_property_list_for_v<empty_properties_t, kp::tag::usm_host>);
static_assert(is_property_list_for_v<empty_properties_t, kp::tag::usm_shared>);
static_assert(is_property_for_v<test_props::host_only, kp::tag::usm_host> &&
              !is_property_for_v<test_props::host_only, kp::tag::usm_shared>);
static_assert(std::is_constructible_v<sycl::usm_allocator<int, alloc::shared>,
                                      sycl::queue, empty_properties_t>);
static_assert(std::is_constructible_v<sycl::usm_allocator<int, alloc::host>,
                                      sycl::context, sycl::device,
                                      test_props::host_only>);
static_assert(!std::is_constructible_v<sycl::usm_allocator<int, alloc::shared>,
                                       sycl::queue, test_props::host_only>);

namespace {
std::map<const void *, ur_usm_type_t> Allocs;

template <typename ParamsT, ur_usm_type_t Type>
ur_result_t afterUSMAlloc(void *pParams) {
  auto Params = *static_cast<ParamsT *>(pParams);
  Allocs[**Params.pppMem] = Type;
  return UR_RESULT_SUCCESS;
}

ur_result_t afterUSMGetMemAllocInfo(void *pParams) {
  auto Params = *static_cast<ur_usm_get_mem_alloc_info_params_t *>(pParams);
  if (*Params.ppropName != UR_USM_ALLOC_INFO_TYPE)
    return UR_RESULT_SUCCESS;
  auto It = Allocs.find(*Params.ppMem);
  if (It == Allocs.end())
    return UR_RESULT_ERROR_INVALID_VALUE;
  if (*Params.ppPropValue)
    *static_cast<ur_usm_type_t *>(*Params.ppPropValue) = It->second;
  return UR_RESULT_SUCCESS;
}

class KhrUSMProperties : public ::testing::Test {
protected:
  void SetUp() override {
    Allocs.clear();
    mock::getCallbacks().set_after_callback(
        "urUSMHostAlloc",
        &afterUSMAlloc<ur_usm_host_alloc_params_t, UR_USM_TYPE_HOST>);
    mock::getCallbacks().set_after_callback(
        "urUSMDeviceAlloc",
        &afterUSMAlloc<ur_usm_device_alloc_params_t, UR_USM_TYPE_DEVICE>);
    mock::getCallbacks().set_after_callback(
        "urUSMSharedAlloc",
        &afterUSMAlloc<ur_usm_shared_alloc_params_t, UR_USM_TYPE_SHARED>);
    mock::getCallbacks().set_after_callback("urUSMGetMemAllocInfo",
                                            &afterUSMGetMemAllocInfo);
  }

  void check(void *Ptr, alloc Kind) {
    ASSERT_NE(Ptr, nullptr);
    EXPECT_EQ(sycl::get_pointer_type(Ptr, Ctx), Kind);
    sycl::free(Ptr, Q);
  }

  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
  sycl::device Dev = Q.get_device();
  sycl::context Ctx = Q.get_context();
};
} // namespace

TEST_F(KhrUSMProperties, DeviceAllocations) {
  properties P{};
  check(sycl::malloc_device(16, Dev, Ctx, P), alloc::device);
  check(sycl::malloc_device<int>(4, Dev, Ctx, empty_properties_t{}),
        alloc::device);
  check(sycl::malloc_device(16, Q, P), alloc::device);
  check(sycl::malloc_device<int>(4, Q, P), alloc::device);
  check(sycl::aligned_alloc_device(64, 16, Dev, Ctx, P), alloc::device);
  check(sycl::aligned_alloc_device<int>(64, 4, Dev, Ctx, P), alloc::device);
  check(sycl::aligned_alloc_device(64, 16, Q, P), alloc::device);
  check(sycl::aligned_alloc_device<int>(64, 4, Q, P), alloc::device);
}

TEST_F(KhrUSMProperties, HostAllocations) {
  properties P{};
  check(sycl::malloc_host(16, Ctx, P), alloc::host);
  check(sycl::malloc_host<int>(4, Ctx, empty_properties_t{}), alloc::host);
  check(sycl::malloc_host(16, Q, P), alloc::host);
  check(sycl::malloc_host<int>(4, Q, P), alloc::host);
  check(sycl::aligned_alloc_host(64, 16, Ctx, P), alloc::host);
  check(sycl::aligned_alloc_host<int>(64, 4, Ctx, P), alloc::host);
  check(sycl::aligned_alloc_host(64, 16, Q, P), alloc::host);
  check(sycl::aligned_alloc_host<int>(64, 4, Q, P), alloc::host);
  check(sycl::malloc_host(16, Q, test_props::host_only{}), alloc::host);
}

TEST_F(KhrUSMProperties, SharedAllocations) {
  properties P{};
  check(sycl::malloc_shared(16, Dev, Ctx, P), alloc::shared);
  check(sycl::malloc_shared<int>(4, Dev, Ctx, empty_properties_t{}),
        alloc::shared);
  check(sycl::malloc_shared(16, Q, P), alloc::shared);
  check(sycl::malloc_shared<int>(4, Q, P), alloc::shared);
  check(sycl::aligned_alloc_shared(64, 16, Dev, Ctx, P), alloc::shared);
  check(sycl::aligned_alloc_shared<int>(64, 4, Dev, Ctx, P), alloc::shared);
  check(sycl::aligned_alloc_shared(64, 16, Q, P), alloc::shared);
  check(sycl::aligned_alloc_shared<int>(64, 4, Q, P), alloc::shared);
}

TEST_F(KhrUSMProperties, ParameterizedAllocations) {
  properties P{};
  for (alloc Kind : {alloc::device, alloc::host, alloc::shared}) {
    check(sycl::malloc(16, Dev, Ctx, Kind, P), Kind);
    check(sycl::malloc<int>(4, Dev, Ctx, Kind, empty_properties_t{}), Kind);
    check(sycl::malloc(16, Q, Kind, P), Kind);
    check(sycl::malloc<int>(4, Q, Kind, P), Kind);
    check(sycl::aligned_alloc(64, 16, Dev, Ctx, Kind, P), Kind);
    check(sycl::aligned_alloc<int>(64, 4, Dev, Ctx, Kind, P), Kind);
    check(sycl::aligned_alloc(64, 16, Q, Kind, P), Kind);
    check(sycl::aligned_alloc<int>(64, 4, Q, Kind, P), Kind);
  }
}

TEST_F(KhrUSMProperties, ParameterizedKindMismatchThrows) {
  properties P{test_props::host_only{}};
  check(sycl::malloc(16, Q, alloc::host, P), alloc::host);
  check(sycl::malloc<int>(4, Dev, Ctx, alloc::host, test_props::host_only{}),
        alloc::host);
  check(sycl::aligned_alloc(64, 16, Q, alloc::host, P), alloc::host);

  for (alloc Kind : {alloc::device, alloc::shared}) {
    auto ExpectInvalid = [](auto &&Alloc) {
      try {
        Alloc();
        FAIL() << "Expected sycl::exception";
      } catch (const sycl::exception &E) {
        EXPECT_EQ(E.code(), sycl::errc::invalid);
      }
    };
    ExpectInvalid([&] { return sycl::malloc(16, Q, Kind, P); });
    ExpectInvalid([&] { return sycl::malloc(16, Dev, Ctx, Kind, P); });
    ExpectInvalid([&] { return sycl::malloc<int>(4, Q, Kind, P); });
    ExpectInvalid([&] {
      return sycl::aligned_alloc<int>(64, 4, Dev, Ctx, Kind,
                                      test_props::host_only{});
    });
  }
  // The throwing calls must not allocate.
  EXPECT_EQ(Allocs.size(), 3u);
}

TEST_F(KhrUSMProperties, Allocator) {
  sycl::usm_allocator<int, alloc::shared> A1(Ctx, Dev, properties{});
  sycl::usm_allocator<int, alloc::host> A2(Q, empty_properties_t{});
  sycl::usm_allocator<int, alloc::host> A3(Q, test_props::host_only{});
  int *P1 = A1.allocate(4);
  int *P2 = A2.allocate(4);
  ASSERT_NE(P1, nullptr);
  ASSERT_NE(P2, nullptr);
  EXPECT_EQ(sycl::get_pointer_type(P1, Ctx), alloc::shared);
  EXPECT_EQ(sycl::get_pointer_type(P2, Ctx), alloc::host);
  A1.deallocate(P1, 4);
  A2.deallocate(P2, 4);
  EXPECT_TRUE(A2 == A3);
}

TEST_F(KhrUSMProperties, OldStyleCalls) {
  // Old-style calls must still resolve unambiguously.
  check(sycl::malloc_device(16, Q), alloc::device);
  check(sycl::malloc_device<int>(4, Q), alloc::device);
  check(sycl::malloc_device(16, Dev, Ctx, sycl::property_list{}),
        alloc::device);
  check(sycl::malloc_host<int>(4, Ctx), alloc::host);
  check(sycl::malloc_shared(16, Q, sycl::property_list{}), alloc::shared);
  check(sycl::aligned_alloc_shared<int>(64, 4, Q), alloc::shared);
  check(sycl::malloc(16, Q, alloc::host), alloc::host);
  check(sycl::malloc<int>(4, Dev, Ctx, alloc::shared), alloc::shared);
  check(sycl::aligned_alloc(64, 16, Q, alloc::device, sycl::property_list{}),
        alloc::device);

  sycl::usm_allocator<int, alloc::shared> A1(Q);
  sycl::usm_allocator<int, alloc::host> A2(Ctx, Dev);
  sycl::usm_allocator<int, alloc::host> A3(Ctx, Dev, sycl::property_list{});
  int *P1 = A1.allocate(2);
  ASSERT_NE(P1, nullptr);
  A1.deallocate(P1, 2);
  EXPECT_TRUE(A2 == A3);
}
