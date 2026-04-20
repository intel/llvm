// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s

#include <sycl/khr/includes/builtins_integer.hpp>
#include <sycl/marray.hpp>
#include <sycl/vector.hpp>

#include <cstdint>
#include <type_traits>

using Vec2i = sycl::vec<std::int32_t, 2>;
using Vec2s = sycl::vec<std::int16_t, 2>;
using Vec2us = sycl::vec<std::uint16_t, 2>;
using MArray2i = sycl::marray<std::int32_t, 2>;
using MArray2s = sycl::marray<std::int16_t, 2>;
using MArray2us = sycl::marray<std::uint16_t, 2>;

// Verify the wrapper exposes scalar, vec, and marray integer builtins.
static_assert(
    std::is_same_v<decltype(sycl::popcount(std::uint32_t{7})), std::uint32_t>);
static_assert(std::is_same_v<decltype(sycl::popcount(Vec2i{1, 3})), Vec2i>);
static_assert(
    std::is_same_v<decltype(sycl::popcount(MArray2i{1, 3})), MArray2i>);
static_assert(
    std::is_same_v<decltype(sycl::mad24(std::int32_t{2}, std::int32_t{3},
                                        std::int32_t{4})),
                   std::int32_t>);
static_assert(
    std::is_same_v<decltype(sycl::mul24(Vec2i{2, 3}, Vec2i{4, 5})), Vec2i>);
static_assert(std::is_same_v<
              decltype(sycl::mul24(MArray2i{2, 3}, MArray2i{4, 5})), MArray2i>);
static_assert(
    std::is_same_v<decltype(sycl::upsample(Vec2s{1, 2}, Vec2us{3, 4})), Vec2i>);
static_assert(
    std::is_same_v<decltype(sycl::upsample(MArray2s{1, 2}, MArray2us{3, 4})),
                   MArray2i>);

int main() {
  // Instantiate representative integer builtin calls from each covered family.
  (void)sycl::popcount(std::uint32_t{7});
  (void)sycl::popcount(Vec2i{1, 3});
  (void)sycl::popcount(MArray2i{1, 3});
  (void)sycl::mad24(std::int32_t{2}, std::int32_t{3}, std::int32_t{4});
  (void)sycl::mul24(Vec2i{2, 3}, Vec2i{4, 5});
  (void)sycl::mul24(MArray2i{2, 3}, MArray2i{4, 5});
  (void)sycl::upsample(Vec2s{1, 2}, Vec2us{3, 4});
  (void)sycl::upsample(MArray2s{1, 2}, MArray2us{3, 4});
  return 0;
}