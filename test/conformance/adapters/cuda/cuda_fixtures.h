#ifndef UR_TEST_CONFORMANCE_ADAPTERS_CUDA_FIXTURES_H_INCLUDED
#define UR_TEST_CONFORMANCE_ADAPTERS_CUDA_FIXTURES_H_INCLUDED
#include <cuda.h>
#include <uur/fixtures.h>

namespace uur {
struct ResultCuda {

    constexpr ResultCuda(CUresult result) noexcept : value(result) {}

    inline bool operator==(const ResultCuda &rhs) const noexcept {
        return rhs.value == value;
    }

    CUresult value;
};
} // namespace uur

#ifndef ASSERT_EQ_RESULT_CUDA
#define ASSERT_EQ_RESULT_CUDA(EXPECTED, ACTUAL)                                \
    ASSERT_EQ(uur::ResultCuda(EXPECTED), uur::ResultCuda(ACTUAL))
#endif // ASSERT_EQ_RESULT_CUDA

#ifndef ASSERT_SUCCESS_CUDA
#define ASSERT_SUCCESS_CUDA(ACTUAL) ASSERT_EQ_RESULT_CUDA(CUDA_SUCCESS, ACTUAL)
#endif // ASSERT_SUCCESS_CUDA

#ifndef EXPECT_EQ_RESULT_CUDA
#define EXPECT_EQ_RESULT_CUDA(EXPECTED, ACTUAL)                                \
    EXPECT_EQ_RESULT_CUDA(uur::ResultCuda(EXPECTED), uur::ResultCuda(ACTUAL))
#endif // EXPECT_EQ_RESULT_CUDA

#ifndef EXPECT_SUCCESS_CUDA
#define EXPECT_SUCCESS_CUDA(ACTUAL)                                            \
    EXPECT_EQ_RESULT_CUDA(UR_RESULT_SUCCESS, ACTUAL)
#endif // EXPECT_EQ_RESULT_CUDA

#endif // UR_TEST_CONFORMANCE_ADAPTERS_CUDA_FIXTURES_H_INCLUDED
