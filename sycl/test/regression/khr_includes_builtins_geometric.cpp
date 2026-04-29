// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s

#include <sycl/khr/includes/builtins_geometric.hpp>
#include <sycl/marray.hpp>
#include <sycl/vector.hpp>

#include <type_traits>

using Vec3f = sycl::vec<float, 3>;
using MArray3f = sycl::marray<float, 3>;

// Verify the wrapper exposes scalar, vec, and marray geometric builtins.
static_assert(std::is_same_v<decltype(sycl::dot(2.0f, 3.0f)), float>);
static_assert(std::is_same_v<decltype(sycl::dot(Vec3f{1.0f, 2.0f, 3.0f},
                                                Vec3f{4.0f, 5.0f, 6.0f})),
                             float>);
static_assert(std::is_same_v<decltype(sycl::dot(MArray3f{1.0f, 2.0f, 3.0f},
                                                MArray3f{4.0f, 5.0f, 6.0f})),
                             float>);
static_assert(std::is_same_v<decltype(sycl::cross(Vec3f{1.0f, 0.0f, 0.0f},
                                                  Vec3f{0.0f, 1.0f, 0.0f})),
                             Vec3f>);
static_assert(std::is_same_v<decltype(sycl::cross(MArray3f{1.0f, 0.0f, 0.0f},
                                                  MArray3f{0.0f, 1.0f, 0.0f})),
                             MArray3f>);
static_assert(
    std::is_same_v<decltype(sycl::normalize(Vec3f{1.0f, 2.0f, 3.0f})), Vec3f>);
static_assert(std::is_same_v<
              decltype(sycl::normalize(MArray3f{1.0f, 2.0f, 3.0f})), MArray3f>);

int main() {
  // Instantiate representative geometric calls that should compile via the
  // wrapper.
  Vec3f A{1.0f, 2.0f, 3.0f};
  Vec3f B{4.0f, 5.0f, 6.0f};
  MArray3f MA{1.0f, 2.0f, 3.0f};
  MArray3f MB{4.0f, 5.0f, 6.0f};
  (void)sycl::dot(2.0f, 3.0f);
  (void)sycl::dot(A, B);
  (void)sycl::dot(MA, MB);
  (void)sycl::cross(Vec3f{1.0f, 0.0f, 0.0f}, Vec3f{0.0f, 1.0f, 0.0f});
  (void)sycl::cross(MArray3f{1.0f, 0.0f, 0.0f}, MArray3f{0.0f, 1.0f, 0.0f});
  (void)sycl::normalize(A);
  (void)sycl::normalize(MA);
  return 0;
}