
//===---- matrix-tensor-cores.hpp - SYCL tensor cores matrix ----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/joint-matrix.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
  namespace ext {
  namespace oneapi {

  namespace detail {

#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
  template <typename S, typename T, size_t NumRows, size_t NumCols,
            sycl::ext::oneapi::experimental::matrix::matrix_use Use,
            sycl::ext::oneapi::experimental::matrix::layout Layout,
            access::address_space Space, typename Cond = void>
  struct load_multiplicand_cuda {
    void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                  S, Use, NumRows, NumCols, Layout, sycl::sub_group> &res,
              multi_ptr<T, Space> src, size_t stride);
  };

  template <sycl::ext::oneapi::experimental::matrix::layout Layout>
  constexpr int get_layout_id();

  template <>
  constexpr int
  get_layout_id<sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
    return 0;
  }

  template <>
  constexpr int
  get_layout_id<sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
    return 1;
  }

#if __cplusplus >= 201703L // if constexpr usage
  template <sycl::ext::oneapi::experimental::matrix::layout LayoutL, typename T,
            size_t NumRows, size_t NumCols, access::address_space Space>
  void load_accumulator_layoutT(
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
          NumRows, NumCols,
          sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          sycl::sub_group> &res,
      multi_ptr<T, Space> src, size_t stride) {
    if constexpr (std::is_same<std::remove_const_t<T>, int32_t>::value) {
      auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        __imma_m16n16k16_ld_c(destptr, src.get(), stride,
                              get_layout_id<LayoutL>());
      } else if constexpr (NumRows == 8 && NumCols == 32) {
        __imma_m8n32k16_ld_c(destptr, src.get(), stride,
                             get_layout_id<LayoutL>());
      } else if constexpr (NumRows == 32 && NumCols == 8) {
        __imma_m32n8k16_ld_c(destptr, src.get(), stride,
                             get_layout_id<LayoutL>());
      }
    } else if constexpr (std::is_same<std::remove_const_t<T>, float>::value) {
      auto dstptr = reinterpret_cast<float *>(&res.wi_marray);
      if constexpr (NumRows == 16 && NumCols == 16) {
        __hmma_m16n16k16_ld_c_f32(dstptr, src.get(), stride,
                                  get_layout_id<LayoutL>());
      } else if constexpr (NumRows == 8 && NumCols == 32) {
        __hmma_m8n32k16_ld_c_f32(dstptr, src.get(), stride,
                                 get_layout_id<LayoutL>());
      } else if constexpr (NumRows == 32 && NumCols == 8) {
        __hmma_m32n8k16_ld_c_f32(dstptr, src.get(), stride,
                                 get_layout_id<LayoutL>());
      }
    } else if constexpr (std::is_same<T, half>::value) {
      auto tileptr = reinterpret_cast<int32_t const *>(src.get());
      auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
      if constexpr (NumRows == 32 && NumCols == 8) {
        __hmma_m32n8k16_ld_c_f16(dstptr, tileptr, stride,
                                 get_layout_id<LayoutL>());
      } else if constexpr (NumRows == 8 && NumCols == 32) {
        __hmma_m8n32k16_ld_c_f16(dstptr, tileptr, stride,
                                 get_layout_id<LayoutL>());
      } else if constexpr (NumRows == 16 && NumCols == 16) {
        __hmma_m16n16k16_ld_c_f16(dstptr, tileptr, stride,
                                  get_layout_id<LayoutL>());
      }
    } else if constexpr (std::is_same<T, double>::value) {
      __dmma_m8n8k4_ld_c(reinterpret_cast<double *>(&res.wi_marray), src.get(),
                         stride, get_layout_id<LayoutL>());
    }
  };
#endif // __cplusplus >= 201703L

  template <typename T, size_t NumRows, size_t NumCols,
            access::address_space Space>
  void load_accumulator_cuda(
      sycl::ext::oneapi::experimental::matrix::joint_matrix<
          T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
          NumRows, NumCols,
          sycl::ext::oneapi::experimental::matrix::layout::dynamic,
          sycl::sub_group> &res,
      multi_ptr<T, Space> src, size_t stride,
      sycl::ext::oneapi::experimental::matrix::layout LayoutAcc) {
    switch (LayoutAcc) {
    case sycl::ext::oneapi::experimental::matrix::layout::row_major:
      load_accumulator_layoutT<
          sycl::ext::oneapi::experimental::matrix::layout::row_major>(res, src,
                                                                      stride);
      break;
    case sycl::ext::oneapi::experimental::matrix::layout::col_major:
      load_accumulator_layoutT<
          sycl::ext::oneapi::experimental::matrix::layout::col_major>(res, src,
                                                                      stride);
      break;
    default:
      assert(false && "Invalid layout specified!");
    }
  }

#if __cplusplus >= 201703L // if constexpr usage
  template <typename S, typename T, size_t NumRows, size_t NumCols,
            sycl::ext::oneapi::experimental::matrix::matrix_use Use,
            sycl::ext::oneapi::experimental::matrix::layout Layout,
            access::address_space Space>
  struct load_multiplicand_cuda<
      S, T, NumRows, NumCols, Use, Layout, Space,
      typename std::enable_if_t<
          Layout ==
              sycl::ext::oneapi::experimental::matrix::layout::row_major ||
          Layout ==
              sycl::ext::oneapi::experimental::matrix::layout::col_major>> {
    void load(sycl::ext::oneapi::experimental::matrix::joint_matrix<
                  S, Use, NumRows, NumCols, Layout, sycl::sub_group> &res,
              multi_ptr<T, Space> src, size_t stride) {
      if constexpr (std::is_same<std::remove_const_t<T>, uint16_t>::value ||
                    std::is_same<
                        std::remove_const_t<T>,
                        sycl::ext::oneapi::experimental::bfloat16>::value) {
        auto tileptr = reinterpret_cast<const int32_t *>(src.get());
        auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
        if constexpr (NumRows == 16 && NumCols == 16) {
          if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                   matrix_use::a) {
            __mma_bf16_m16n16k16_ld_a(destptr, tileptr, stride,
                                      get_layout_id<Layout>());
          } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                          matrix_use::b) {
            __mma_bf16_m16n16k16_ld_b(destptr, tileptr, stride,
                                      get_layout_id<Layout>());
          }
        } else if constexpr (NumRows == 8 && NumCols == 16) {
          __mma_bf16_m8n32k16_ld_a(destptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 32) {
          __mma_bf16_m8n32k16_ld_b(destptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (NumRows == 32 && NumCols == 16) {
          __mma_bf16_m32n8k16_ld_a(destptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 8) {
          __mma_bf16_m32n8k16_ld_b(destptr, tileptr, stride,
                                   get_layout_id<Layout>());
        }
      } else if constexpr (std::is_same<std::remove_const_t<T>,
                                        uint8_t>::value) {
        auto tileptr = reinterpret_cast<const int32_t *>(src.get());
        auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
        if constexpr (NumRows == 16 && NumCols == 16) {
          if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                   matrix_use::a) {
            __imma_m16n16k16_ld_a_u8(destptr, tileptr, stride,
                                     get_layout_id<Layout>());
          } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                          matrix_use::b) {
            __imma_m16n16k16_ld_b_u8(destptr, tileptr, stride,
                                     get_layout_id<Layout>());
          }
        } else if constexpr (NumRows == 8 && NumCols == 16) {
          __imma_m8n32k16_ld_a_u8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 32) {
          __imma_m8n32k16_ld_b_u8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        } else if constexpr (NumRows == 32 && NumCols == 16) {
          __imma_m32n8k16_ld_a_u8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 8) {
          __imma_m32n8k16_ld_b_u8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        }
      } else if constexpr (std::is_same<std::remove_const_t<T>,
                                        int8_t>::value) {
        auto tileptr = reinterpret_cast<const int32_t *>(src.get());
        auto destptr = reinterpret_cast<int32_t *>(&res.wi_marray);
        if constexpr (NumRows == 16 && NumCols == 16) {
          if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                   matrix_use::a) {
            __imma_m16n16k16_ld_a_s8(destptr, tileptr, stride,
                                     get_layout_id<Layout>());
          } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                          matrix_use::b) {
            __imma_m16n16k16_ld_b_s8(destptr, tileptr, stride,
                                     get_layout_id<Layout>());
          }
        } else if constexpr (NumRows == 8 && NumCols == 16) {
          __imma_m8n32k16_ld_a_s8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 32) {
          __imma_m8n32k16_ld_b_s8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        } else if constexpr (NumRows == 32 && NumCols == 16) {
          __imma_m32n8k16_ld_a_s8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 8) {
          __imma_m32n8k16_ld_b_s8(destptr, tileptr, stride,
                                  get_layout_id<Layout>());
        }
      } else if constexpr (std::is_same<std::remove_const_t<T>, half>::value) {
        auto tileptr = reinterpret_cast<const int32_t *>(src.get());
        auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
        if constexpr (NumRows == 16 && NumCols == 16) {
          if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                   matrix_use::a) {
            __hmma_m16n16k16_ld_a(dstptr, tileptr, stride,
                                  get_layout_id<Layout>());
          } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                          matrix_use::b) {
            __hmma_m16n16k16_ld_b(dstptr, tileptr, stride,
                                  get_layout_id<Layout>());
          }
        } else if constexpr (NumRows == 8 && NumCols == 16) {
          __hmma_m8n32k16_ld_a(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 32) {
          __hmma_m8n32k16_ld_b(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
        } else if constexpr (NumRows == 32 && NumCols == 16) {
          __hmma_m32n8k16_ld_a(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
        } else if constexpr (NumRows == 16 && NumCols == 8) {
          __hmma_m32n8k16_ld_b(dstptr, tileptr, stride,
                               get_layout_id<Layout>());
        }

      } else if constexpr (std::is_same<S,
                                        sycl::ext::oneapi::experimental::
                                            matrix::precision::tf32>::value) {
        auto tileptr = reinterpret_cast<const int32_t *>(src.get());
        auto dstptr = reinterpret_cast<int32_t *>(&res.wi_marray);
        if constexpr (NumRows == 16 && NumCols == 8) {
          __mma_tf32_m16n16k8_ld_a(dstptr, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (NumRows == 8 && NumCols == 16) {
          __mma_tf32_m16n16k8_ld_b(dstptr, tileptr, stride,
                                   get_layout_id<Layout>());
        }
      } else if constexpr (std::is_same<std::remove_const_t<T>,
                                        double>::value) {
        auto dstptr = reinterpret_cast<double *>(&res.wi_marray);
        if constexpr (Use ==
                      sycl::ext::oneapi::experimental::matrix::matrix_use::a) {
          __dmma_m8n8k4_ld_a(dstptr, src.get(), stride,
                             get_layout_id<Layout>());
        } else if constexpr (Use == sycl::ext::oneapi::experimental::matrix::
                                        matrix_use::b) {
          __dmma_m8n8k4_ld_b(dstptr, src.get(), stride,
                             get_layout_id<Layout>());
        }
      }
    }
  };
#endif // __cplusplus >= 201703L

#if __cplusplus >= 201703L // if constexpr usage
  template <typename T, size_t NumRows, size_t NumCols,
            access::address_space Space>
  struct joint_matrix_store_cuda_impl {
    template <sycl::ext::oneapi::experimental::matrix::layout LayoutL>
    void storeLayoutT(
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
            NumRows, NumCols,
            sycl::ext::oneapi::experimental::matrix::layout::dynamic,
            sycl::sub_group> &src,
        multi_ptr<T, Space> dst, size_t stride) {
      if constexpr (NumRows == 16 && NumCols == 16) {
        if constexpr (std::is_same<T, float>::value) {
          __hmma_m16n16k16_st_c_f32(dst.get(),
                                    reinterpret_cast<float *>(&src.wi_marray),
                                    stride, get_layout_id<LayoutL>());
        } else if constexpr (std::is_same<T, int32_t>::value) {
          __imma_m16n16k16_st_c_i32(dst.get(),
                                    reinterpret_cast<int32_t *>(&src.wi_marray),
                                    stride, get_layout_id<LayoutL>());
        } else if constexpr (std::is_same<T, half>::value) {
          __hmma_m16n16k16_st_c_f16(reinterpret_cast<int32_t *>(dst.get()),
                                    reinterpret_cast<int32_t *>(&src.wi_marray),
                                    stride, get_layout_id<LayoutL>());
        }
      } else if constexpr (NumRows == 8 && NumCols == 32) {
        if constexpr (std::is_same<T, float>::value) {
          __hmma_m8n32k16_st_c_f32(dst.get(),
                                   reinterpret_cast<float *>(&src.wi_marray),
                                   stride, get_layout_id<LayoutL>());
        } else if constexpr (std::is_same<T, int32_t>::value) {
          __imma_m8n32k16_st_c_i32(dst.get(),
                                   reinterpret_cast<int32_t *>(&src.wi_marray),
                                   stride, get_layout_id<LayoutL>());
        } else if constexpr (std::is_same<T, half>::value) {
          __hmma_m8n32k16_st_c_f16(reinterpret_cast<int32_t *>(dst.get()),
                                   reinterpret_cast<int32_t *>(&src.wi_marray),
                                   stride, get_layout_id<LayoutL>());
        }
      } else if constexpr (NumRows == 32 && NumCols == 8) {
        if constexpr (std::is_same<T, float>::value) {
          __hmma_m32n8k16_st_c_f32(dst.get(),
                                   reinterpret_cast<float *>(&src.wi_marray),
                                   stride, get_layout_id<LayoutL>());
        } else if constexpr (std::is_same<T, int32_t>::value) {
          __imma_m32n8k16_st_c_i32(dst.get(),
                                   reinterpret_cast<int32_t *>(&src.wi_marray),
                                   stride, get_layout_id<LayoutL>());
        } else if constexpr (std::is_same<T, half>::value) {
          __hmma_m32n8k16_st_c_f16(reinterpret_cast<int32_t *>(dst.get()),
                                   reinterpret_cast<int32_t *>(&src.wi_marray),
                                   stride, get_layout_id<LayoutL>());
        }
      } else if constexpr (std::is_same<T, double>::value) {
        __dmma_m8n8k4_st_c_f64(dst.get(),
                               reinterpret_cast<double *>(&src.wi_marray),
                               stride, get_layout_id<LayoutL>());
      }
    }
    void store(
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            T, sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator,
            NumRows, NumCols,
            sycl::ext::oneapi::experimental::matrix::layout::dynamic,
            sycl::sub_group> &src,
        multi_ptr<T, Space> dst, size_t stride,
        sycl::ext::oneapi::experimental::matrix::layout LayoutAcc) {
      switch (LayoutAcc) {
      case sycl::ext::oneapi::experimental::matrix::layout::row_major:
        storeLayoutT<
            sycl::ext::oneapi::experimental::matrix::layout::row_major>(
            src, dst, stride);
        break;
      case sycl::ext::oneapi::experimental::matrix::layout::col_major:
        storeLayoutT<
            sycl::ext::oneapi::experimental::matrix::layout::col_major>(
            src, dst, stride);
        break;
      default:
        assert(false && "Invalid layout specified!");
      }
    }
  };
#endif // __cplusplus >= 201703L

  template <typename Tm, typename Tc, typename Td, std::size_t M, std::size_t K,
            std::size_t N,
            sycl::ext::oneapi::experimental::matrix::layout LayoutA,
            sycl::ext::oneapi::experimental::matrix::layout LayoutB,
            typename Cond = void>
  struct joint_matrix_mad_cuda_impl {
    void
    mad(sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Td,
            sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
            N, sycl::ext::oneapi::experimental::matrix::layout::dynamic,
            sycl::sub_group> &D,
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Tm, sycl::ext::oneapi::experimental::matrix::matrix_use::a, M, K,
            LayoutA, sycl::sub_group> &A,
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Tm, sycl::ext::oneapi::experimental::matrix::matrix_use::b, K, N,
            LayoutB, sycl::sub_group> &B,
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Tc,
            sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
            N, sycl::ext::oneapi::experimental::matrix::layout::dynamic,
            sycl::sub_group> &C);
  };

  template <sycl::ext::oneapi::experimental::matrix::layout LayoutA,
            sycl::ext::oneapi::experimental::matrix::layout LayoutB>
  constexpr int get_layout_pair_id();

  template <>
  constexpr int get_layout_pair_id<
      sycl::ext::oneapi::experimental::matrix::layout::row_major,
      sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
    return 0;
  }

  template <>
  constexpr int get_layout_pair_id<
      sycl::ext::oneapi::experimental::matrix::layout::row_major,
      sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
    return 1;
  }

  template <>
  constexpr int get_layout_pair_id<
      sycl::ext::oneapi::experimental::matrix::layout::col_major,
      sycl::ext::oneapi::experimental::matrix::layout::row_major>() {
    return 2;
  }

  template <>
  constexpr int get_layout_pair_id<
      sycl::ext::oneapi::experimental::matrix::layout::col_major,
      sycl::ext::oneapi::experimental::matrix::layout::col_major>() {
    return 3;
  }

#if __cplusplus >= 201703L // if constexpr usage
  template <typename Tm, typename Tc, typename Td, std::size_t M, std::size_t K,
            std::size_t N,
            sycl::ext::oneapi::experimental::matrix::layout LayoutA,
            sycl::ext::oneapi::experimental::matrix::layout LayoutB>
  struct joint_matrix_mad_cuda_impl<
      Tm, Tc, Td, M, K, N, LayoutA, LayoutB,
      typename std::enable_if_t<
          (LayoutA ==
               sycl::ext::oneapi::experimental::matrix::layout::row_major ||
           LayoutA ==
               sycl::ext::oneapi::experimental::matrix::layout::col_major) &&
          (LayoutB ==
               sycl::ext::oneapi::experimental::matrix::layout::row_major ||
           LayoutB ==
               sycl::ext::oneapi::experimental::matrix::layout::col_major)>> {
    void
    mad(sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Td,
            sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
            N, sycl::ext::oneapi::experimental::matrix::layout::dynamic,
            sycl::sub_group> &D,
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Tm, sycl::ext::oneapi::experimental::matrix::matrix_use::a, M, K,
            LayoutA, sycl::sub_group> &A,
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Tm, sycl::ext::oneapi::experimental::matrix::matrix_use::b, K, N,
            LayoutB, sycl::sub_group> &B,
        sycl::ext::oneapi::experimental::matrix::joint_matrix<
            Tc,
            sycl::ext::oneapi::experimental::matrix::matrix_use::accumulator, M,
            N, sycl::ext::oneapi::experimental::matrix::layout::dynamic,
            sycl::sub_group> &C) {
      if constexpr (M == 16 && N == 16 && K == 16) {
        if constexpr (std::is_same<Tc, int32_t>::value) {
          auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
          auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
          auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
          auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
          if constexpr (std::is_same<Tm, int8_t>::value) {
            __imma_m16n16k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                    get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<Tm, uint8_t>::value) {
            __imma_m16n16k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                    get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        } else if constexpr (std::is_same<Tm, half>::value) {
          auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
          auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
          if constexpr (std::is_same<Tc, float>::value) {
            if constexpr (std::is_same<Td, float>::value) {
              __hmma_m16n16k16_mma_f32f32(
                  reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
                  reinterpret_cast<const float *>(&C.wi_marray),
                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
            } else {
              __hmma_m16n16k16_mma_f16f32(
                  reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
                  reinterpret_cast<float const *>(&C.wi_marray),
                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
            }
          } else if constexpr (std::is_same<Tc, half>::value) {
            if constexpr (std::is_same<Td, float>::value) {
              __hmma_m16n16k16_mma_f32f16(
                  reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
                  reinterpret_cast<int const *>(&C.wi_marray),
                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
            } else {
              __hmma_m16n16k16_mma_f16f16(
                  reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
                  reinterpret_cast<const int32_t *>(&C.wi_marray),
                  get_layout_pair_id<LayoutA, LayoutB>(), 0);
            }
          }
        } else if constexpr (std::is_same<Tm, uint16_t>::value ||
                             std::is_same<Tm, sycl::ext::oneapi::experimental::
                                                  bfloat16>::value) {
          __mma_bf16_m16n16k16_mma_f32(
              reinterpret_cast<float *>(&D.wi_marray),
              reinterpret_cast<const int32_t *>(&A.wi_marray),
              reinterpret_cast<const int32_t *>(&B.wi_marray),
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (M == 8 && N == 32 && K == 16) {
        if constexpr (std::is_same<Tc, int32_t>::value) {
          auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
          auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
          auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
          auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
          if constexpr (std::is_same<Tm, int8_t>::value) {
            __imma_m8n32k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<Tm, uint8_t>::value) {
            __imma_m8n32k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        } else if constexpr (std::is_same<Tm, half>::value) {
          auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
          auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
          if constexpr (std::is_same<Tc, float>::value) {
            __hmma_m8n32k16_mma_f32f32(
                reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
                reinterpret_cast<const float *>(&C.wi_marray),
                get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<Tc, half>::value) {
            __hmma_m8n32k16_mma_f16f16(
                reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
                reinterpret_cast<const int32_t *>(&C.wi_marray),
                get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        } else if constexpr (std::is_same<Tm, uint16_t>::value ||
                             std::is_same<Tm, sycl::ext::oneapi::experimental::
                                                  bfloat16>::value) {
          __mma_bf16_m8n32k16_mma_f32(
              reinterpret_cast<float *>(&D.wi_marray),
              reinterpret_cast<const int32_t *>(&A.wi_marray),
              reinterpret_cast<const int32_t *>(&B.wi_marray),
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        }
      } else if constexpr (M == 32 && N == 8 && K == 16) {
        if constexpr (std::is_same<Tc, int32_t>::value) {
          auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
          auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
          auto ptrC = reinterpret_cast<const int32_t *>(&C.wi_marray);
          auto ptrD = reinterpret_cast<int32_t *>(&D.wi_marray);
          if constexpr (std::is_same<Tm, int8_t>::value) {
            __imma_m32n8k16_mma_s8(ptrD, ptrA, ptrB, ptrC,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<Tm, uint8_t>::value) {
            __imma_m32n8k16_mma_u8(ptrD, ptrA, ptrB, ptrC,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        } else if constexpr (std::is_same<Tm, uint16_t>::value ||
                             std::is_same<Tm, sycl::ext::oneapi::experimental::
                                                  bfloat16>::value) {
          __mma_bf16_m32n8k16_mma_f32(
              reinterpret_cast<float *>(&D.wi_marray),
              reinterpret_cast<const int32_t *>(&A.wi_marray),
              reinterpret_cast<const int32_t *>(&B.wi_marray),
              reinterpret_cast<const float *>(&C.wi_marray),
              get_layout_pair_id<LayoutA, LayoutB>(), 0);
        } else if constexpr (std::is_same<Tm, half>::value) {
          auto ptrA = reinterpret_cast<const int32_t *>(&A.wi_marray);
          auto ptrB = reinterpret_cast<const int32_t *>(&B.wi_marray);
          if constexpr (std::is_same<Tc, float>::value) {
            __hmma_m32n8k16_mma_f32f32(
                reinterpret_cast<float *>(&D.wi_marray), ptrA, ptrB,
                reinterpret_cast<const float *>(&C.wi_marray),
                get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<Tc, half>::value) {
            __hmma_m32n8k16_mma_f16f16(
                reinterpret_cast<int32_t *>(&D.wi_marray), ptrA, ptrB,
                reinterpret_cast<const int32_t *>(&C.wi_marray),
                get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        }
      } else if constexpr (M == 16 && N == 16 && K == 8) {
        __mma_tf32_m16n16k8_mma_f32(reinterpret_cast<float *>(&D.wi_marray),
                                    reinterpret_cast<int32_t *>(&A.wi_marray),
                                    reinterpret_cast<int32_t *>(&B.wi_marray),
                                    reinterpret_cast<float *>(&C.wi_marray),
                                    get_layout_pair_id<LayoutA, LayoutB>(), 0);
      } else if constexpr (std::is_same<Tm, double>::value) {
        __dmma_m8n8k4_mma_f64(reinterpret_cast<double *>(&D.wi_marray),
                              reinterpret_cast<const double *>(&A.wi_marray),
                              reinterpret_cast<const double *>(&B.wi_marray),
                              reinterpret_cast<const double *>(&C.wi_marray),
                              get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }
    }
  };
#endif // __cplusplus >= 201703L
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)

  } // namespace detail
  } // namespace oneapi
  } // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

