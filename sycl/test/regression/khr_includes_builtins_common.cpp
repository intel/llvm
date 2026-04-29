// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s

#include <sycl/khr/includes/builtins_common.hpp>
#include <sycl/marray.hpp>
#include <sycl/vector.hpp>

#include <type_traits>

using Vec2f = sycl::vec<float, 2>;
using MArray2f = sycl::marray<float, 2>;

// Verify the wrapper exposes scalar, half, vec, and marray common builtins.
static_assert(std::is_same_v<decltype(sycl::degrees(1.0f)), float>);
static_assert(
    std::is_same_v<decltype(sycl::degrees(sycl::half{1.0f})), sycl::half>);
static_assert(std::is_same_v<
              decltype(sycl::clamp(Vec2f{0.25f, 1.5f}, 0.0f, 1.0f)), Vec2f>);
static_assert(
    std::is_same_v<decltype(sycl::clamp(MArray2f{0.25f, 1.5f}, 0.0f, 1.0f)),
                   MArray2f>);
static_assert(std::is_same_v<decltype(sycl::mix(Vec2f{0.0f, 1.0f},
                                                Vec2f{1.0f, 2.0f}, 0.5f)),
                             Vec2f>);
static_assert(std::is_same_v<decltype(sycl::mix(MArray2f{0.0f, 1.0f},
                                                MArray2f{1.0f, 2.0f}, 0.5f)),
                             MArray2f>);
static_assert(
    std::is_same_v<decltype(sycl::step(0.5f, Vec2f{0.25f, 0.75f})), Vec2f>);

int main() {
  // Instantiate the same families of calls to check the wrapper is usable.
  Vec2f A{0.0f, 1.0f};
  Vec2f B{1.0f, 2.0f};
  MArray2f MA{0.0f, 1.0f};
  MArray2f MB{1.0f, 2.0f};
  (void)sycl::degrees(1.0f);
  (void)sycl::degrees(sycl::half{1.0f});
  (void)sycl::clamp(Vec2f{0.25f, 1.5f}, 0.0f, 1.0f);
  (void)sycl::clamp(MArray2f{0.25f, 1.5f}, 0.0f, 1.0f);
  (void)sycl::mix(A, B, 0.5f);
  (void)sycl::mix(MA, MB, 0.5f);
  (void)sycl::step(0.5f, Vec2f{0.25f, 0.75f});
  return 0;
}