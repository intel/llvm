// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsyntax-only -fpreview-breaking-changes %s

#include "sycl/access/access.hpp"
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;

static_assert(std::is_nothrow_copy_constructible_v<accessor<int, 1>>,
              "Copy constructor must be noexcept");
static_assert(std::is_nothrow_copy_assignable_v<accessor<int, 1>>,
              "Copy assignment must be noexcept");
static_assert(std::is_nothrow_move_constructible_v<accessor<int, 1>>,
              "Move constructor must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<accessor<int, 1>>,
              "Move assignment must be noexcept");

static_assert(std::is_nothrow_copy_constructible_v<local_accessor<int, 1>>,
              "Copy constructor must be noexcept");
static_assert(std::is_nothrow_copy_assignable_v<local_accessor<int, 1>>,
              "Copy assignment must be noexcept");
static_assert(std::is_nothrow_move_constructible_v<local_accessor<int, 1>>,
              "Move constructor must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<local_accessor<int, 1>>,
              "Move assignment must be noexcept");

static_assert(std::is_nothrow_copy_constructible_v<host_accessor<int, 1>>,
              "Copy constructor must be noexcept");
static_assert(std::is_nothrow_copy_assignable_v<host_accessor<int, 1>>,
              "Copy assignment must be noexcept");
static_assert(std::is_nothrow_move_constructible_v<host_accessor<int, 1>>,
              "Move constructor must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<host_accessor<int, 1>>,
              "Move assignment must be noexcept");

static_assert(
    std::is_nothrow_copy_constructible_v<
        accessor<int, 1, access::mode::read_write, access::target::local>>,
    "Copy constructor must be noexcept");
static_assert(
    std::is_nothrow_copy_assignable_v<
        accessor<int, 1, access::mode::read_write, access::target::local>>,
    "Copy assignment must be noexcept");
static_assert(
    std::is_nothrow_move_constructible_v<
        accessor<int, 1, access::mode::read_write, access::target::local>>,
    "Move constructor must be noexcept");
static_assert(
    std::is_nothrow_move_assignable_v<
        accessor<int, 1, access::mode::read_write, access::target::local>>,
    "Move assignment must be noexcept");

static_assert(std::is_nothrow_copy_constructible_v<
                  unsampled_image_accessor<float4, 1, access::mode::read>>,
              "Copy constructor must be noexcept");
static_assert(std::is_nothrow_copy_assignable_v<
                  unsampled_image_accessor<float4, 1, access::mode::read>>,
              "Copy assignment must be noexcept");
static_assert(std::is_nothrow_move_constructible_v<
                  unsampled_image_accessor<float4, 1, access::mode::read>>,
              "Move constructor must be noexcept");
static_assert(std::is_nothrow_move_assignable_v<
                  unsampled_image_accessor<float4, 1, access::mode::read>>,
              "Move assignment must be noexcept");

static_assert(
    std::is_nothrow_copy_constructible_v<sampled_image_accessor<float4, 1>>,
    "Copy constructor must be noexcept");
static_assert(
    std::is_nothrow_copy_assignable_v<sampled_image_accessor<float4, 1>>,
    "Copy assignment must be noexcept");
static_assert(
    std::is_nothrow_move_constructible_v<sampled_image_accessor<float4, 1>>,
    "Move constructor must be noexcept");
static_assert(
    std::is_nothrow_move_assignable_v<sampled_image_accessor<float4, 1>>,
    "Move assignment must be noexcept");

int main() {
  queue Q;
  buffer<int, 1> Buf{range<1>{1}};

  Q.submit([&](handler &CGH) {
    accessor Acc{Buf, CGH};
    using AccT = accessor<int, 1, access_mode::read_write, target::device>;
    static_assert(std::is_same_v<decltype(Acc), AccT>);

    // Copy, move, and prvalue deduction must all yield the source type.
    accessor AccCopy(Acc);
    static_assert(std::is_same_v<decltype(AccCopy), AccT>);
    accessor AccMove(std::move(AccCopy));
    static_assert(std::is_same_v<decltype(AccMove), AccT>);
    accessor AccPrvalue{accessor{Buf, CGH}};
    static_assert(std::is_same_v<decltype(AccPrvalue), AccT>);

    local_accessor<int, 1> LocalAcc{range<1>{1}, CGH};
    local_accessor LocalAccCopy(LocalAcc);
    static_assert(std::is_same_v<decltype(LocalAccCopy), decltype(LocalAcc)>);
    local_accessor LocalAccMove(std::move(LocalAccCopy));
    static_assert(std::is_same_v<decltype(LocalAccMove), decltype(LocalAcc)>);

    accessor AccProps{
        Buf, CGH, ext::oneapi::accessor_property_list{ext::oneapi::no_alias}};
    accessor AccPropsCopy(AccProps);
    static_assert(std::is_same_v<decltype(AccPropsCopy), decltype(AccProps)>);
  });

  host_accessor HostAcc{Buf};
  host_accessor HostAccCopy(HostAcc);
  static_assert(std::is_same_v<decltype(HostAccCopy), decltype(HostAcc)>);
  host_accessor HostAccMove(std::move(HostAccCopy));
  static_assert(std::is_same_v<decltype(HostAccMove), decltype(HostAcc)>);

  return 0;
}