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

void matrix_multiply_ref(bfloat16 *A, bfloat16 *B, float *C, int MATRIX_M,
                         int MATRIX_N, int MATRIX_K, bool transpose_c = false) {
  for (unsigned int i = 0; i < MATRIX_M; i++) {
    for (unsigned int k = 0; k < MATRIX_K; k++) {
      for (unsigned int j = 0; j < MATRIX_N; j++) {
        int c_ind = transpose_c ? (j * MATRIX_M + i) : i * MATRIX_N + j;
        C[c_ind] +=
            make_fp32(A[i * MATRIX_K + k]) * make_fp32(B[k * MATRIX_N + j]);
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
  bool res = true;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if ((fabs(src[i * cols + j] - (T1)ref[i * cols + j])) > BF16_EPSILON) {
        res = false;
      }
    }
  }
  return res;
}
