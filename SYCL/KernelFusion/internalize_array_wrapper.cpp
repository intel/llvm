// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || hip
// REQUIRES: fusion

// Test internalization of a nested array type.

#include <array>

#include <sycl/sycl.hpp>

using namespace sycl;

template <size_t N, size_t M> struct array_wrapper {
  static constexpr size_t rows{N};
  static constexpr size_t columns{M};
  static constexpr size_t vec_width{2};

  using value_type = vec<int, vec_width>;
  using reference_type = value_type &;
  using const_reference_type = const value_type &;

  std::array<std::array<value_type, columns>, rows> vs;

  explicit array_wrapper(const_reference_type v) {
    std::array<value_type, columns> el;
    el.fill(v);
    vs.fill(el);
  }

  array_wrapper() : array_wrapper{value_type{}} {}

  constexpr std::array<value_type, columns> &operator[](size_t i) {
    return vs[i];
  }

  constexpr const std::array<value_type, columns> &operator[](size_t i) const {
    return vs[i];
  }
};

int main() {
  constexpr size_t dataSize = 2;
  constexpr size_t rows = 2;
  constexpr size_t columns = 2;

  using array_type = array_wrapper<rows, columns>;

  array_type in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize],
      out[dataSize];

  for (size_t id = 0; id < dataSize; ++id) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < columns; ++j) {
        in1[id][i][j].s0() = in1[id][i][j].s1() = id * 2;
        in2[id][i][j].s0() = in2[id][i][j].s1() = id * 3;
        in3[id][i][j].s0() = in3[id][i][j].s1() = id * 4;
        tmp[id][i][j].s0() = tmp[id][i][j].s1() = -1;
        out[id][i][j].s0() = out[id][i][j].s1() = -1;
      }
    }
  }

  queue q{default_selector_v,
          {ext::codeplay::experimental::property::queue::enable_fusion{}}};

  {
    buffer<array_type> bIn1{in1, range{dataSize}};
    buffer<array_type> bIn2{in2, range{dataSize}};
    buffer<array_type> bIn3{in3, range{dataSize}};
    buffer<array_type> bTmp{tmp, range{dataSize}};
    buffer<array_type> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(dataSize, [=](id<1> id) {
        const auto &accIn1Wrapp = accIn1[id];
        const auto &accIn2Wrapp = accIn2[id];
        auto &accTmpWrapp = accTmp[id];
        for (size_t i = 0; i < dataSize; ++i) {
          const auto &in1 = accIn1Wrapp[i];
          const auto &in2 = accIn2Wrapp[i];
          auto &tmp = accTmpWrapp[i];
          for (size_t j = 0; j < columns; ++j) {
            tmp[j] = in1[j] + in2[j];
          }
        }
      });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(dataSize, [=](id<1> id) {
        const auto &tmpWrapp = accTmp[id];
        const auto &accIn3Wrapp = accIn3[id];
        auto &accOutWrapp = accOut[id];
        for (size_t i = 0; i < dataSize; ++i) {
          const auto &tmp = tmpWrapp[i];
          const auto &in3 = accIn3Wrapp[i];
          auto &out = accOutWrapp[i];
          for (size_t j = 0; j < columns; ++j) {
            out[j] = tmp[j] * in3[j];
          }
        }
      });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  constexpr array_type::value_type not_written{-1, -1};
  for (size_t id = 0; id < dataSize; ++id) {
    const array_type::value_type expected{20 * id * id, 20 * id * id};
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < columns; ++j) {
        assert(all(out[id][i][j] == expected) && "Computation error");
        assert(all(tmp[id][i][j] == not_written) && "Not internalizing");
      }
    }
  }

  return 0;
}
