#include <sycl/sycl.hpp>

using bfloat16 = sycl::ext::oneapi::bfloat16;

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
                         int MATRIX_N, int MATRIX_K) {
  for (unsigned int i = 0; i < MATRIX_M; i++) {
    for (unsigned int k = 0; k < MATRIX_K; k++) {
      for (unsigned int j = 0; j < MATRIX_N; j++) {
        C[i * MATRIX_N + j] +=
            make_fp32(A[i * MATRIX_K + k]) * make_fp32(B[k * MATRIX_N + j]);
      }
    }
  }
}

void matrix_multiply_ref_transposed_c(bfloat16 *A, bfloat16 *B, float *C,
                                      int MATRIX_M, int MATRIX_N,
                                      int MATRIX_K) {
  for (unsigned int i = 0; i < MATRIX_M; i++) {
    for (unsigned int k = 0; k < MATRIX_K; k++) {
      for (unsigned int j = 0; j < MATRIX_N; j++) {
        C[j * MATRIX_M + i] +=
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
