// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -fpreview-breaking-changes %s

// Checks that the move constructor and the move assignment operator of every
// class with common reference semantics (SYCL 2020 section 3.5.2) and common
// by-value semantics (section 3.5.3) are declared noexcept, as required by
// tables 3 and 5.

#include <sycl/sycl.hpp>

#include <memory>
#include <type_traits>

template <typename T> constexpr bool declaresMoveAssign(T &(T::*)(T &&)) {
  return true;
}

#define CHECK_MOVE_NOEXCEPT(...)                                               \
  static_assert(std::is_nothrow_move_constructible_v<__VA_ARGS__>,             \
                #__VA_ARGS__ " must be nothrow move constructible");           \
  static_assert(std::is_nothrow_move_assignable_v<__VA_ARGS__>,                \
                #__VA_ARGS__ " must be nothrow move assignable");              \
  static_assert(declaresMoveAssign<__VA_ARGS__>(&__VA_ARGS__::operator=),      \
                #__VA_ARGS__ " must declare operator=(" #__VA_ARGS__ " &&)");

// --- Common reference semantics (SYCL 2020 section 3.5.2) -------------------

CHECK_MOVE_NOEXCEPT(sycl::platform)
CHECK_MOVE_NOEXCEPT(sycl::device)
CHECK_MOVE_NOEXCEPT(sycl::context)
CHECK_MOVE_NOEXCEPT(sycl::queue)
CHECK_MOVE_NOEXCEPT(sycl::event)
CHECK_MOVE_NOEXCEPT(sycl::kernel)
CHECK_MOVE_NOEXCEPT(sycl::kernel_id)
CHECK_MOVE_NOEXCEPT(sycl::stream)

CHECK_MOVE_NOEXCEPT(sycl::kernel_bundle<sycl::bundle_state::input>)
CHECK_MOVE_NOEXCEPT(sycl::kernel_bundle<sycl::bundle_state::object>)
CHECK_MOVE_NOEXCEPT(sycl::kernel_bundle<sycl::bundle_state::executable>)

CHECK_MOVE_NOEXCEPT(sycl::device_image<sycl::bundle_state::input>)
CHECK_MOVE_NOEXCEPT(sycl::device_image<sycl::bundle_state::object>)
CHECK_MOVE_NOEXCEPT(sycl::device_image<sycl::bundle_state::executable>)

CHECK_MOVE_NOEXCEPT(sycl::buffer<int, 1>)
CHECK_MOVE_NOEXCEPT(sycl::buffer<int, 2>)
CHECK_MOVE_NOEXCEPT(sycl::buffer<int, 3>)
CHECK_MOVE_NOEXCEPT(sycl::buffer<int, 1, std::allocator<int>>)

CHECK_MOVE_NOEXCEPT(sycl::unsampled_image<1>)
CHECK_MOVE_NOEXCEPT(sycl::unsampled_image<2>)
CHECK_MOVE_NOEXCEPT(sycl::unsampled_image<3>)
CHECK_MOVE_NOEXCEPT(sycl::sampled_image<1>)
CHECK_MOVE_NOEXCEPT(sycl::sampled_image<2>)
CHECK_MOVE_NOEXCEPT(sycl::sampled_image<3>)

CHECK_MOVE_NOEXCEPT(
    sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::target::device>)
CHECK_MOVE_NOEXCEPT(
    sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::device>)
CHECK_MOVE_NOEXCEPT(
    sycl::accessor<int, 2, sycl::access_mode::write, sycl::target::device>)
CHECK_MOVE_NOEXCEPT(
    sycl::accessor<int, 3, sycl::access_mode::read_write, sycl::target::device>)
CHECK_MOVE_NOEXCEPT(sycl::accessor<int, 1, sycl::access_mode::read_write,
                                   sycl::target::host_task>)

CHECK_MOVE_NOEXCEPT(sycl::local_accessor<int, 0>)
CHECK_MOVE_NOEXCEPT(sycl::local_accessor<int, 1>)
CHECK_MOVE_NOEXCEPT(sycl::local_accessor<int, 3>)

CHECK_MOVE_NOEXCEPT(sycl::host_accessor<int, 0>)
CHECK_MOVE_NOEXCEPT(sycl::host_accessor<int, 1>)
CHECK_MOVE_NOEXCEPT(sycl::host_accessor<int, 3>)

CHECK_MOVE_NOEXCEPT(
    sycl::unsampled_image_accessor<sycl::int4, 1, sycl::access_mode::read>)
CHECK_MOVE_NOEXCEPT(
    sycl::unsampled_image_accessor<sycl::int4, 3, sycl::access_mode::write>)
CHECK_MOVE_NOEXCEPT(sycl::host_unsampled_image_accessor<sycl::int4, 1>)
CHECK_MOVE_NOEXCEPT(sycl::host_unsampled_image_accessor<sycl::int4, 3>)
CHECK_MOVE_NOEXCEPT(sycl::sampled_image_accessor<sycl::float4, 1>)
CHECK_MOVE_NOEXCEPT(sycl::sampled_image_accessor<sycl::float4, 3>)
CHECK_MOVE_NOEXCEPT(sycl::host_sampled_image_accessor<sycl::float4, 1>)
CHECK_MOVE_NOEXCEPT(sycl::host_sampled_image_accessor<sycl::float4, 3>)

// --- Common by-value semantics (SYCL 2020 section 3.5.3) -------------------

CHECK_MOVE_NOEXCEPT(sycl::id<1>)
CHECK_MOVE_NOEXCEPT(sycl::id<2>)
CHECK_MOVE_NOEXCEPT(sycl::id<3>)
CHECK_MOVE_NOEXCEPT(sycl::range<1>)
CHECK_MOVE_NOEXCEPT(sycl::range<2>)
CHECK_MOVE_NOEXCEPT(sycl::range<3>)
CHECK_MOVE_NOEXCEPT(sycl::item<1, true>)
CHECK_MOVE_NOEXCEPT(sycl::item<1, false>)
CHECK_MOVE_NOEXCEPT(sycl::item<3, true>)
CHECK_MOVE_NOEXCEPT(sycl::item<3, false>)
CHECK_MOVE_NOEXCEPT(sycl::nd_item<1>)
CHECK_MOVE_NOEXCEPT(sycl::nd_item<3>)
CHECK_MOVE_NOEXCEPT(sycl::h_item<1>)
CHECK_MOVE_NOEXCEPT(sycl::h_item<3>)
CHECK_MOVE_NOEXCEPT(sycl::group<1>)
CHECK_MOVE_NOEXCEPT(sycl::group<3>)
CHECK_MOVE_NOEXCEPT(sycl::sub_group)
CHECK_MOVE_NOEXCEPT(sycl::nd_range<1>)
CHECK_MOVE_NOEXCEPT(sycl::nd_range<3>)
