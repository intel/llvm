#include <cmath>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

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

template <typename Ta, typename Tc>
void matrix_multiply_ref(Ta *A, Ta *B, Tc *C, int M, int N, int K,
                         bool transpose_c = false) {
  for (unsigned int m = 0; m < M; m++) {
    for (unsigned int n = 0; n < N; n++) {
      for (unsigned int k = 0; k < K; k++) {
        int c_ind = transpose_c ? (n * M + m) : m * N + n;
        if constexpr (std::is_same_v<Ta, bfloat16> && std::is_same_v<Tc, float>)
          C[c_ind] += make_fp32(A[m * K + k]) * make_fp32(B[k * N + n]);
        else if constexpr (std::is_same_v<Ta, float> &&
                               std::is_same_v<Tc, float> ||
                           std::is_same_v<Ta, int8_t> &&
                               std::is_same_v<Tc, int32_t>)
          C[c_ind] += A[m * K + k] * B[k * N + n];
        else
          assert(false && "Unsupported type in matrix_multiply_ref.");
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

template <typename T1, typename T2>
bool matrix_compare(unsigned int rows, unsigned int cols, T1 *src, T2 *ref) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if constexpr (std::is_same_v<T1, float> || std::is_same_v<T1, bfloat16>) {
        float diff = std::fabs(src[i * cols + j] - (T1)ref[i * cols + j]);
        if (diff > FLOAT_EPSILON) {
          std::cout << "Incorrect result in matrix. Ref: "
                    << (T1)ref[i * cols + j] << ", Val: " << src[i * cols + j]
                    << ", Diff: " << diff << ", Epsilon: " << FLOAT_EPSILON
                    << "\n";
          return false;
        }
      } else if constexpr (std::is_same_v<T1, int32_t>) {
        if (src[i * cols + j] != ref[i * cols + j]) {
          std::cout << "Incorrect result in matrix. Ref: " << ref[i * cols + j]
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
