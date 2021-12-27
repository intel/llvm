#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental::matrix {

enum class matrix_use { a, b, accumulator };

enum class matrix_layout { row_major, col_major, packed_a, packed_b };

template <typename T, matrix_use MT, size_t Rows = sycl::dynamic_extent,
          size_t Cols = sycl::dynamic_extent,
          matrix_layout Layout = matrix_layout::row_major,
          typename Group = sycl::sub_group, typename Cond = void>
struct joint_matrix {
  joint_matrix(Group g) {}
};

// The enable_if_t usage in this file is used to disable the
// matrix_layout::packed case which is not compatible with the Nvidia cuda
// backend.
template <matrix_layout Layout>
struct joint_matrix<
    double, matrix_use::a, 8, 4, Layout, sycl::sub_group,
    typename std::enable_if_t<Layout == matrix_layout::row_major ||
                              Layout == matrix_layout::col_major>> {
  double data[1];
};

template <matrix_layout Layout>
struct joint_matrix<
    double, matrix_use::b, 4, 8, Layout, sycl::sub_group,
    typename std::enable_if_t<(Layout == matrix_layout::row_major ||
                               Layout == matrix_layout::col_major)>> {
  double data[1];
};

template <matrix_layout Layout>
struct joint_matrix<
    double, matrix_use::accumulator, 8, 8, Layout, sycl::sub_group,
    typename std::enable_if_t<Layout == matrix_layout::row_major ||
                              Layout == matrix_layout::col_major>> {
  double data[2];
};

} // namespace experimental::matrix

namespace detail {

template <typename T, sycl::ext::oneapi::experimental::matrix::matrix_use MT,
          size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space, typename Cond = void>
struct joint_matrix_load_impl {
  void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                T, MT, NumRows, NumCols, Layout> &res,
            multi_ptr<T, Space> src, size_t stride);
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout Layout>
constexpr int get_layout_id();

template <>
constexpr int get_layout_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major>() {
  return 0;
}

template <>
constexpr int get_layout_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major>() {
  return 1;
}

template <sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space>
struct joint_matrix_load_impl<
    double, sycl::ext::oneapi::experimental::matrix::matrix_use::a, 8, 4,
    Layout, Space,
    typename std::enable_if_t<Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::row_major ||
                              Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::col_major>> {
  void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                double, sycl::ext::oneapi::experimental::matrix::matrix_use::a,
                8, 4, Layout> &res,
            multi_ptr<double, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __dmma_m8n8k4_ld_a(res.data, src.get(), stride, get_layout_id<Layout>());
#endif
#endif
  }
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space>
struct joint_matrix_load_impl<
    double, sycl::ext::oneapi::experimental::matrix::matrix_use::b, 4, 8,
    Layout, Space,
    typename std::enable_if_t<Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::row_major ||
                              Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::col_major>> {
  void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                double, sycl::ext::oneapi::experimental::matrix::matrix_use::b,
                4, 8, Layout> &res,
            multi_ptr<double, Space> src, size_t stride) {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __dmma_m8n8k4_ld_b(res.data, src.get(), stride, get_layout_id<Layout>());
#endif
#endif
  }
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space>
struct joint_matrix_load_impl<
    double, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, 8,
    8, Layout, Space,
    typename std::enable_if_t<Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::row_major ||
                              Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::col_major>> {
  void
  load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
           double,
           sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, 8,
           8, Layout> &res,
       multi_ptr<double, Space> src, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __dmma_m8n8k4_ld_c(res.data, src.get(), stride, get_layout_id<Layout>());
#endif
#endif
  }
};

template <typename T, size_t NumRows, size_t NumCols,
          sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space, typename Cond = void>
struct joint_matrix_store_impl {
  void
  store(sycl::ext::oneapi::experimental::matrix::joint_matrix<
            T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
            NumRows, NumCols, Layout> &src,
        multi_ptr<T, Space> dst, size_t stride);
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout Layout,
          access::address_space Space>
struct joint_matrix_store_impl<
    double, 8, 8, Layout, Space,
    typename std::enable_if_t<Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::row_major ||
                              Layout == sycl::ext::oneapi::experimental::
                                            matrix::matrix_layout::col_major>> {
  void
  store(sycl::ext::oneapi::experimental::matrix::joint_matrix<
            double,
            sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, 8,
            8, Layout> &src,
        multi_ptr<double, Space> dst, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride,
                           get_layout_id<Layout>());
#endif
#endif
  }
};

template <typename T1, typename T2, std::size_t M, std::size_t K, std::size_t N,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutB,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutC,
          typename Cond = void>
struct joint_matrix_mad_impl {
  sycl::ext::oneapi::experimental::matrix::joint_matrix<
      T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
      N, LayoutC>
  mad(sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T1, sycl::ext::oneapi::experimental::matrix::matrix_use::a, M, K,
          LayoutA>
          A,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T1, sycl::ext::oneapi::experimental::matrix::matrix_use::b, K, N,
          LayoutB>
          B,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T2, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
          M, N, LayoutC>
          C);
};

template <sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutB>
constexpr int get_layout_pair_id();

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major>() {
  return 0;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major>() {
  return 1;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::row_major>() {
  return 2;
}

template <>
constexpr int get_layout_pair_id<
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major,
    sycl::ext::oneapi::experimental::matrix::matrix_layout::col_major>() {
  return 3;
}

template <sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutA,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutB,
          sycl::ext::oneapi::experimental::matrix::matrix_layout LayoutC>
struct joint_matrix_mad_impl<
    double, double, 8, 4, 8, LayoutA, LayoutB, LayoutC,
    typename std::enable_if_t<
        (LayoutA == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        row_major ||
         LayoutA == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        col_major) &&
        (LayoutB == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        row_major ||
         LayoutB == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        col_major) &&
        (LayoutC == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        row_major ||
         LayoutC == sycl::ext::oneapi::experimental::matrix::matrix_layout::
                        col_major)>> {
  sycl::ext::oneapi::experimental::matrix::joint_matrix<
      double, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
      8, 8, LayoutC>
  mad(sycl::ext::oneapi::experimental::matrix::joint_matrix<
          double, sycl::ext::oneapi::experimental::matrix::matrix_use::a, 8, 4,
          LayoutA>
          A,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          double, sycl::ext::oneapi::experimental::matrix::matrix_use::b, 4, 8,
          LayoutB>
          B,
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          double,
          sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, 8,
          8, LayoutC>
          C) {
    sycl::ext::oneapi::experimental::matrix::joint_matrix<
        double,
        sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, 8, 8,
        LayoutC>
        D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data,
                          get_layout_pair_id<LayoutA, LayoutB>(), 0);
#endif
#endif

    return D;
  }
};

} // namespace detail

namespace experimental::matrix {

template <typename Group, typename T, matrix_use MT, size_t NumRows,
          size_t NumCols, matrix_layout Layout, access::address_space Space>
void joint_matrix_load(
    Group sg, joint_matrix<T, MT, NumRows, NumCols, Layout, Group> &res,
    multi_ptr<T, Space> src, size_t stride) {
  sycl::ext::oneapi::detail::joint_matrix_load_impl<T, MT, NumRows, NumCols,
                                                    Layout, Space>{}
      .load(res, src, stride);
}

template <typename Group, typename T, size_t NumRows, size_t NumCols,
          matrix_layout Layout, access::address_space Space>
void joint_matrix_store(Group sg,
                        joint_matrix<T, matrix_use::accumulator, NumRows,
                                     NumCols, Layout, Group> &src,
                        multi_ptr<T, Space> dst, size_t stride) {
  sycl::ext::oneapi::detail::joint_matrix_store_impl<T, NumRows, NumCols,
                                                     Layout, Space>{}
      .store(src, dst, stride);
}

template <typename Group, typename T1, typename T2, std::size_t M,
          std::size_t K, std::size_t N, matrix_layout LayoutA,
          matrix_layout LayoutB, matrix_layout LayoutC>
joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group>
joint_matrix_mad(
    Group sg, joint_matrix<T1, matrix_use::a, M, K, LayoutA, Group> A,
    joint_matrix<T1, matrix_use::b, K, N, LayoutB, Group> B,
    joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group> C) {
  return sycl::ext::oneapi::detail::joint_matrix_mad_impl<
             T1, T2, M, K, N, LayoutA, LayoutB, LayoutC>{}
      .mad(A, B, C);
}

} // namespace experimental::matrix
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
