### Using DPAS matrix multiply-add-and-accumulate operations

In this example, we demonstrate how to use Dot Product Accumulate Systolic
APIs, or `dpas`.

Compile and run:
```bash
> clang++ -fsycl dpas.cpp

> ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./a.out
```

Source code:
```C++

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

inline auto create_exception_handler() {
  return [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << "sycl::exception: " << e0.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "std::exception: " << e.what() << std::endl;
      } catch (...) {
        std::cout << "generic exception\n";
      }
    }
  };
}

struct usm_deleter {
  queue q;
  void operator()(void *ptr) {
    if (ptr)
      sycl::free(ptr, q);
  }
};

// Res = A * B.
// Assume the HW is PVC.

constexpr int SystolicDepth = 8;
constexpr int RepeatCount = 4;
constexpr int ExecSize = 16; // 16 for PVC, 8 for DG2.

// Let A and B be matrices of unsigned 4-bit integers.
constexpr xmx::dpas_argument_type BPrec = xmx::dpas_argument_type::u4;
constexpr xmx::dpas_argument_type APrec = xmx::dpas_argument_type::u4;

constexpr int AElemBitSize = 4; // 4-bit integers.
constexpr int BElemBitSize = 4; // 4-bit integers.

// Elements of A and B will are packed into uint8_t,
// meaning that one uint8_t holds two 4-bit unsigned integers.
// Packaging for A and res is horizontal, for B is vertical.
using PackedType = unsigned char;
using APackedType = PackedType;
using BPackedType = PackedType;

// Result type, according to documentation is either int or uint.
using ResType = unsigned int; // as both A and B are unsigned.

constexpr int OpsPerChannel =
    std::min(32 / std::max(AElemBitSize, BElemBitSize), 8);

// A(M x K) * B(K x N) + C(M x N).
// where:
constexpr int M = RepeatCount;
constexpr int K = SystolicDepth * OpsPerChannel;
constexpr int N = ExecSize;

void write_to_horizontally_packed_matrix_a(APackedType *vec, int row, int col,
                                           APackedType value) {
  constexpr int NumRows = M;
  constexpr int NumCols = K;

  constexpr int elems_in_elem_t = sizeof(APackedType) * 8 / AElemBitSize;
  int unpacked_linear_index = row * NumCols + col;
  int packed_linear_index = unpacked_linear_index / elems_in_elem_t;

  // Assume that we deal only with 2 or 4-bit integers.
  static_assert((AElemBitSize == 2 || AElemBitSize == 4),
                "Unexpected element type");

  // Update the corresponding bits of the target element.
  APackedType target_elem = vec[packed_linear_index];
  // target_elem has 2 or more elements in it. Need to extract one.
  unsigned int offset =
      (unpacked_linear_index % elems_in_elem_t) * AElemBitSize;
  unsigned int mask = (1 << AElemBitSize) - 1;
  value = (value & mask) << offset;
  mask = mask << offset;
  target_elem = (target_elem & ~mask) | value;
  vec[packed_linear_index] = target_elem;
}

APackedType read_from_horizontally_packed_matrix_a(APackedType *vec, int row,
                                                   int col) {
  constexpr int NumRows = M;
  constexpr int NumCols = K;

  // Assume that we deal only with 2 or 4-bit integers.
  static_assert((AElemBitSize == 2 || AElemBitSize == 4),
                "Unexpected element type");
  // Assume the packed type is unsigned.
  static_assert(std::is_unsigned_v<APackedType>, "Expect unsigned packed type");

  // 1. Find and read the target 'unsigned int' element.
  // The unpacked matrix has dimensions: NumRows*NumCols
  constexpr int elems_in_elem_t = sizeof(APackedType) * 8 / AElemBitSize;
  int unpacked_linear_index = row * NumCols + col;
  int packed_linear_index = unpacked_linear_index / elems_in_elem_t;
  APackedType target_elem = vec[packed_linear_index];

  // 2. Extract, add sign and return the value.
  // target_elem has 2 or more elements in it. Need to extract one.
  unsigned int offset =
      (unpacked_linear_index % elems_in_elem_t) * AElemBitSize;
  unsigned int mask = (static_cast<uint64_t>(1) << AElemBitSize) - 1;
  APackedType value = (target_elem >> offset) & mask;
  return value;
}

void write_to_vertically_packed_matrix_b(BPackedType *vec, int row, int col,
                                         BPackedType value) {
  constexpr int NumRows = K;
  constexpr int NumCols = N;

  // 1. Find and read the target 'int' element.
  // The unpacked matrix has dimensions: NumRows*NumCols.
  constexpr int ElemsInInt = 32 / BElemBitSize;
  int packed_row = row / ElemsInInt;
  int packed_linear_index = packed_row * NumCols + col;
  int target_elem = vec[packed_linear_index];

  // 2. Insert sub-element 'value' into 32-bit int and write back to matrix.
  int elem_bit_offset = (row % ElemsInInt) * BElemBitSize;
  int mask = (static_cast<uint64_t>(1) << BElemBitSize) - 1;
  int i_value = sycl::bit_cast<BPackedType>(value);
  i_value = (i_value & mask) << elem_bit_offset;
  mask = mask << elem_bit_offset;
  target_elem = (target_elem & ~mask) | i_value;
  vec[packed_linear_index] = target_elem;
}

BPackedType read_from_vertically_packed_matrix_b(BPackedType *vvec, int row,
                                                 int col) {
  constexpr int NumRows = K;
  constexpr int NumCols = N;

  int *vec = reinterpret_cast<int *>(vvec);

  // Assume that we deal only with 2 or 4-bit integers.
  static_assert((BElemBitSize == 2 || BElemBitSize == 4),
                "Unexpected element type");
  // Assume the packed type is unsigned.
  static_assert(std::is_unsigned_v<BPackedType>, "Expect unsigned packed type");

  // Find and read the target 'int' element.
  // The unpacked matrix has dimensions: NumRows*NumCols.
  constexpr int ElemsInInt = 32 / BElemBitSize;

  int packed_row = row / ElemsInInt;
  int target_elem = vec[packed_row * NumCols + col];

  // 2. Extract the queried sub-elem from 32-bit int, bit-cast to BPackedType
  // and return.
  int elem_bit_offset = (row % ElemsInInt) * BElemBitSize;
  unsigned int mask = (static_cast<uint64_t>(1) << BElemBitSize) - 1;
  int value = (target_elem >> elem_bit_offset) & mask;
  return static_cast<BPackedType>(value);
}

int main() {
  unsigned n_errs = 0;
  try {
    queue q(gpu_selector_v, create_exception_handler());
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>()
              << std::endl;

    constexpr unsigned Size = 128;
    constexpr unsigned VL = 16;

    constexpr int APackedSize =
        M * K * AElemBitSize / (sizeof(APackedType) * 8);
    constexpr int BPackedSize =
        K * N * BElemBitSize / (sizeof(BPackedType) * 8);

    auto a_packed = aligned_alloc_shared<APackedType>(128, APackedSize, q);
    auto b_packed = aligned_alloc_shared<BPackedType>(128, BPackedSize, q);
    auto res = aligned_alloc_shared<ResType>(128, M * N, q);

    std::unique_ptr<APackedType, usm_deleter> guard_a(a_packed, usm_deleter{q});
    std::unique_ptr<BPackedType, usm_deleter> guard_b(b_packed, usm_deleter{q});
    std::unique_ptr<ResType, usm_deleter> guard_res(res, usm_deleter{q});

    // Initialize a_packed;
    unsigned value = 0;
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        value += 1;
        write_to_horizontally_packed_matrix_a(a_packed, i, j,
                                              static_cast<APackedType>(value));
      }
    }

    // Initialize b_packed;
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < N; j++) {
        int value = (i + j % 4) == 0 ? 1 : (2 + i + j) % 3;
        write_to_vertically_packed_matrix_b(b_packed, i, j,
                                            static_cast<BPackedType>(value));
        assert(value == (int)(static_cast<BPackedType>(value)) && "ERROR");
      }
    }

    q.single_task([=]() [[intel::sycl_explicit_simd]] {
       esimd::simd<APackedType, APackedSize> a(a_packed,
                                               esimd::overaligned_tag<16>{});
       esimd::simd<BPackedType, BPackedSize> b(b_packed,
                                               esimd::overaligned_tag<16>{});
       esimd::simd<ResType, M * N> c;

       // Compute C = AxB;
       c = xmx::dpas<8, RepeatCount, ResType, BPackedType, APackedType, BPrec,
                     APrec>(b, a);
       c.copy_to(res);
     }).wait();

    // Verify with HOST computation.
    auto a = a_packed;
    auto b = b_packed;
    for (int i = 0; i < M && n_errs < 10; i++) {
      for (int j = 0; j < N && n_errs < 10; j++) {
        ResType gold_res = 0;

        // res(i,j) = C(i,j) = A(i,*)*B(*,j))
        for (int k = 0; k < K; k++) {
          APackedType a_val = read_from_horizontally_packed_matrix_a(a, i, k);
          BPackedType b_val = read_from_vertically_packed_matrix_b(b, k, j);
          gold_res += a_val * b_val;
        }
        // res(i,j) is res[N*i + j]
        if (res[N * i + j] != gold_res) {
          n_errs++;
          std::cerr << "res[" << i << ", " << j << "] = (" << res[M * i + j]
                    << ") != expected (" << gold_res << ")" << std::endl;
        }
      }
    }
  } catch (sycl::exception &e) {
    std::cerr << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }

  std::cerr << ((n_errs > 0) ? "FAILED" : "PASSED") << std::endl;
  return (n_errs > 0) ? 1 : 0;
}
```
