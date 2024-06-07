// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out

// Test complete fusion with internalization of a deep struct type.

#include <type_traits>

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

struct deep_vec {
  using value_type = vec<int, 4>;
  struct level_0 {
    struct level_1 {
      struct level_2 {
        deep_vec::value_type v;

        constexpr level_2() = default;
        constexpr explicit level_2(const deep_vec::value_type &v) : v{v} {}
      } v;
      constexpr level_1() = default;
      constexpr explicit level_1(const deep_vec::value_type &v) : v{v} {}
    } v;

    constexpr level_0() = default;
    constexpr explicit level_0(const deep_vec::value_type &v) : v{v} {}
  } v;

  constexpr deep_vec() = default;
  constexpr explicit deep_vec(const value_type &v) : v{v} {}

  constexpr value_type &operator*() { return v.v.v.v; }
  constexpr value_type *operator->() { return &this->operator*(); }
};

deep_vec operator+(deep_vec lhs, deep_vec rhs) { return deep_vec{*lhs + *rhs}; }
deep_vec operator*(deep_vec lhs, deep_vec rhs) { return deep_vec{*lhs * *rhs}; }

int main() {
  constexpr size_t dataSize = 512;

  deep_vec in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize],
      out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i]->s0() = in1[i]->s1() = in1[i]->s2() = in1[i]->s3() = i * 2;
    in2[i]->s0() = in2[i]->s1() = in2[i]->s2() = in2[i]->s3() = i * 3;
    in3[i]->s0() = in3[i]->s1() = in3[i]->s2() = in3[i]->s3() = i * 4;
    tmp[i]->s0() = tmp[i]->s1() = tmp[i]->s2() = tmp[i]->s3() = -1;
    out[i]->s0() = out[i]->s1() = out[i]->s2() = out[i]->s3() = -1;
  }

  queue q{default_selector_v,
          {ext::codeplay::experimental::property::queue::enable_fusion{}}};

  {
    buffer<deep_vec> bIn1{in1, range{dataSize}};
    buffer<deep_vec> bIn2{in2, range{dataSize}};
    buffer<deep_vec> bIn3{in3, range{dataSize}};
    buffer<deep_vec> bTmp{tmp, range{dataSize}};
    buffer<deep_vec> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(
          dataSize, [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  constexpr deep_vec::value_type not_written{-1, -1, -1, -1};
  for (size_t i = 0; i < dataSize; ++i) {
    const deep_vec::value_type expected{20 * i * i, 20 * i * i, 20 * i * i,
                                        20 * i * i};
    assert(all(*out[i] == expected) && "Computation error");
    assert(all(*tmp[i] == not_written) && "Not internalizing");
  };

  return 0;
}
