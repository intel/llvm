//==------------------ common.hpp  - DPC++ joint_matrix---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cmath>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
namespace syclintelex = sycl::ext::intel::experimental;
using bfloat16 = sycl::ext::oneapi::bfloat16;

// Most of the time, failures related to floating-point calculations (both float
// and bfloat16) are caused by accumulation errors rather than the algorithm
// itself. If it is an algorithm issue, the calculated result gap from the
// reference would be much bigger. To avoid flaky test results while catching
// algorithm errors, we are increasing the accuracy threshold.
// Something like this should be good enough to catch algorithm errors:
// fabs(ref[i] - val[i])/max(fabs(ref)) < 10e-2
constexpr float FLOAT_EPSILON = 10e-2;

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

float make_fp32(bfloat16 x) {
  unsigned int y = *((int *)&x);
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

template <typename Ta, typename Tb, typename Tc, unsigned int VF = 1>
void matrix_multiply_ref(Ta *A, Tb *B, Tc *C, int M, int N, int K,
                         bool transpose_c = false, bool colmajor_a = false,
                         bool colmajor_b = false) {
  for (unsigned int m = 0; m < M; m++) {
    for (unsigned int n = 0; n < N; n++) {
      for (unsigned int k = 0; k < K; k++) {

        int a_ind = colmajor_a ? (k * M + m) : m * K + k;
        int b_ind = colmajor_b ? (n * K + k) : k * N + n;
        int c_ind = transpose_c ? (n * M + m) : m * N + n;

        Ta *va = (Ta *)(A + a_ind * VF);
        Tb *vb = (Tb *)(B + b_ind * VF);
        Tc acc = *(C + c_ind);

        for (unsigned int i = 0; i < VF; i++) {
          if constexpr (std::is_same_v<Ta, bfloat16> &&
                        std::is_same_v<Tc, float>)
            acc += make_fp32(va[i]) * make_fp32(vb[i]);
          else if constexpr (std::is_same_v<Ta, float> &&
                                 std::is_same_v<Tc, float> ||
                             std::is_integral_v<Ta> && std::is_integral_v<Tc>)
            acc += va[i] * vb[i];
          else if constexpr (std::is_same_v<Ta, sycl::half> &&
                             std::is_same_v<Tc, float>)
            acc += (float)va[i] * (float)vb[i];
          else
            assert(false && "Unsupported type in matrix_multiply_ref.");
        }

        *(C + c_ind) = acc;
      }
    }
  }
}

template <typename T>
void matrix_vnni(unsigned int rows, unsigned int cols, T *src, T *dest,
                 unsigned int vnniFactor = 2) {
  for (unsigned int i = 0; i < rows / vnniFactor; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int k = 0; k < vnniFactor; k++) {
        dest[i * cols * vnniFactor + j * vnniFactor + k] =
            src[(i * vnniFactor + k) * cols + j];
      }
    }
  }
}

template <typename T>
void matrix_transpose(unsigned int rows, unsigned int cols, T *dst, T *src) {
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      dst[i + j * rows] = src[i * cols + j];
    }
  }
}

template <typename T>
void matrix_fill(unsigned int rows, unsigned int cols, T *src, T val) {
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      src[i * cols + j] = val;
    }
  }
}

template <typename T, typename F>
void matrix_fill(unsigned int rows, unsigned int cols, T *src, F op) {
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      src[i * cols + j] = T(op(i, j));
    }
  }
}

template <typename T>
void matrix_rand(unsigned int rows, unsigned int cols, T *src, T val) {
  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-val, val);
  std::uniform_int_distribution idistr((int)-val, (int)val);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      if constexpr (std::is_same_v<T, bfloat16> || std::is_same_v<T, float>) {
        src[i * cols + j] = T(fdistr(dev));
      } else if constexpr (std::is_same_v<T, int8_t> ||
                           std::is_same_v<T, int32_t>) {
        src[i * cols + j] = T(idistr(dev));
      } else {
        assert(false && "Unsupported type in matrix_rand.");
      }
    }
  }
}

template <typename T>
void matrix_copy(unsigned int rows, unsigned int cols, T *src, T *dst) {
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      dst[i * cols + j] = src[i * cols + j];
    }
  }
}

template <typename T1, typename T2, bool exact = false>
bool matrix_compare(unsigned int rows, unsigned int cols, T1 *src, T2 *ref) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if constexpr (!exact && (std::is_same_v<T1, float> ||
                               std::is_same_v<T1, bfloat16> ||
                               (std::is_same_v<T1, double> &&
                                std::is_same_v<T2, double>))) {
        float diff = std::fabs(src[i * cols + j] - (T1)ref[i * cols + j]);
        if (diff > FLOAT_EPSILON || std::isnan(src[i * cols + j])) {
          std::cout << "Incorrect result in matrix. "
                    << "i: " << i << ", j: " << j
                    << ", Ref: " << (T1)ref[i * cols + j]
                    << ", Val: " << src[i * cols + j] << ", Diff: " << diff
                    << ", Epsilon: " << FLOAT_EPSILON << "\n";
          return false;
        }
      } else if constexpr (exact || std::is_same_v<T1, int32_t>) {
        if (src[i * cols + j] != ref[i * cols + j]) {
          std::cout << "Incorrect result in matrix." << "i: " << i
                    << ", j: " << j << ", Ref: " << ref[i * cols + j]
                    << ", Val: " << src[i * cols + j] << "\n";
          return false;
        }
      } else {
        std::cout << "Unsupported type in matrix_compare\n";
        return false;
      }
    }
  }
  return true;
}

bool is_type_supported_by_device(queue q, matrix_type type) {
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  for (int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == type) {
      return true;
    }
  }
  return false;
}

template <typename KernelName> size_t get_sg_size(queue q) {
  auto KernelID = get_kernel_id<KernelName>();
  auto KB =
      get_kernel_bundle<bundle_state::executable>(q.get_context(), {KernelID});
  auto kernel = KB.get_kernel(KernelID);

  return kernel
      .template get_info<info::kernel_device_specific::max_sub_group_size>(
          q.get_device());
}
