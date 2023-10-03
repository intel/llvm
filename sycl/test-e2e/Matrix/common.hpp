#include <cmath>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using bfloat16 = sycl::ext::oneapi::bfloat16;

constexpr float BF16_EPSILON = 0.00781250;
constexpr float FLOAT_EPSILON = 10e-3;

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
        if (std::is_same_v<Ta, bfloat16> && std::is_same_v<Tc, float>)
          C[c_ind] += make_fp32(A[m * K + k]) * make_fp32(B[k * N + n]);
        if (std::is_same_v<Ta, int8_t> && std::is_same_v<Tc, int32_t>)
          C[c_ind] += A[m * K + k] * B[k * N + n];
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

template <typename T>
void matrix_rand(unsigned int rows, unsigned int cols, T *src, T val) {
  std::random_device dev;
  std::uniform_real_distribution<float> fdistr(-val, val);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      src[i * cols + j] = T(fdistr(dev));
    }
  }
}

template <typename T1, typename T2>
bool matrix_compare(unsigned int rows, unsigned int cols, T1 *src, T2 *ref) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if constexpr (std::is_same_v<T1, float> || std::is_same_v<T1, bfloat16>) {
        if (std::fabs(src[i * cols + j] - (T1)ref[i * cols + j]) >
            BF16_EPSILON) {
          std::cout << "Incorrect result in matrix\n";
          return false;
        }
      } else if (std::is_same_v<T1, int32_t>) {
        if (src[i * cols + j] != ref[i * cols + j]) {
          std::cout << "Incorrect result in matrix\n";
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
