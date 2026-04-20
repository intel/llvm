// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s

#include <sycl/khr/includes/builtins_relational.hpp>
#include <sycl/marray.hpp>
#include <sycl/vector.hpp>

#include <cstdint>
#include <type_traits>

using Vec2f = sycl::vec<float, 2>;
using Vec2i = sycl::vec<std::int32_t, 2>;
using MArray2f = sycl::marray<float, 2>;
using MArray2b = sycl::marray<bool, 2>;

// Verify the wrapper exposes scalar, half, vec, and marray relational builtins.
static_assert(std::is_same_v<decltype(sycl::islessgreater(1.0f, 2.0f)), bool>);
static_assert(std::is_same_v<decltype(sycl::islessgreater(sycl::half{1.0f},
                                                          sycl::half{2.0f})),
                             bool>);
static_assert(
    std::is_same_v<
        decltype(sycl::isequal(Vec2f{1.0f, 2.0f}, Vec2f{1.0f, 3.0f})), Vec2i>);
static_assert(std::is_same_v<decltype(sycl::isequal(MArray2f{1.0f, 2.0f},
                                                    MArray2f{1.0f, 3.0f})),
                             MArray2b>);
static_assert(
    std::is_same_v<decltype(sycl::signbit(Vec2f{-1.0f, 2.0f})), Vec2i>);
static_assert(std::is_same_v<decltype(sycl::signbit(sycl::half{-1.0f})), bool>);
static_assert(
    std::is_same_v<decltype(sycl::select(Vec2f{1.0f, 2.0f}, Vec2f{3.0f, 4.0f},
                                         Vec2i{0, -1})),
                   Vec2f>);
static_assert(std::is_same_v<decltype(sycl::select(MArray2f{1.0f, 2.0f},
                                                   MArray2f{3.0f, 4.0f},
                                                   MArray2b{false, true})),
                             MArray2f>);

int main() {
  // Instantiate representative relational calls and mask-producing operations.
  Vec2f A{1.0f, 2.0f};
  Vec2f B{3.0f, 4.0f};
  Vec2i Mask{0, -1};
  MArray2f MA{1.0f, 2.0f};
  MArray2f MB{3.0f, 4.0f};
  MArray2b MMask{false, true};
  (void)sycl::islessgreater(1.0f, 2.0f);
  (void)sycl::islessgreater(sycl::half{1.0f}, sycl::half{2.0f});
  (void)sycl::isequal(A, B);
  (void)sycl::isequal(MA, MB);
  (void)sycl::signbit(Vec2f{-1.0f, 2.0f});
  (void)sycl::signbit(sycl::half{-1.0f});
  (void)sycl::select(A, B, Mask);
  (void)sycl::select(MA, MB, MMask);
  return 0;
}