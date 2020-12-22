//==------------ esimd_libcm.hpp libcm interface definitions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ESIMD_LIBCM_HPP_
#define _ESIMD_LIBCM_HPP_

#ifndef __SYCL_DEVICE_ONLY__

#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/host_profiling_info.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/range.hpp>

#include <thread>

#include "esimdcpu_runtime.h" // TODO : From CM library

#include "esimd_libcm_lambda_wrapper.hpp"

#define _COMMA_ ,

LAMBDA_WRAPPER_TMPL(sycl::id<3>, ID_3DIM)
LAMBDA_WRAPPER_TMPL(sycl::id<2>, ID_2DIM)
LAMBDA_WRAPPER_TMPL(sycl::id<1>, ID_1DIM)
LAMBDA_WRAPPER_TMPL(sycl::item<3 _COMMA_ false>, ITEM_3DIM)
LAMBDA_WRAPPER_TMPL(sycl::item<2 _COMMA_ false>, ITEM_2DIM)
LAMBDA_WRAPPER_TMPL(sycl::item<1 _COMMA_ false>, ITEM_1DIM)
LAMBDA_WRAPPER_TMPL(sycl::item<3 _COMMA_ true>, ITEM_OFFSET_3DIM)
LAMBDA_WRAPPER_TMPL(sycl::item<2 _COMMA_ true>, ITEM_OFFSET_2DIM)
LAMBDA_WRAPPER_TMPL(sycl::item<1 _COMMA_ true>, ITEM_OFFSET_1DIM)
LAMBDA_WRAPPER_TMPL(sycl::nd_item<3>, NDITEM_3DIM)
LAMBDA_WRAPPER_TMPL(sycl::nd_item<2>, NDITEM_2DIM)
LAMBDA_WRAPPER_TMPL(sycl::nd_item<1>, NDITEM_1DIM)

template <class KernelType, class KernelArgType, typename IteratorType,
          int __Dims__>
class libCMBatch {
private:
  KernelType MKernel;
  std::vector<IteratorType> argVector;
  std::vector<uint32_t> spaceDim;
  const std::vector<uint32_t> singleGrpDim = {1, 1, 1};
  const uint32_t hwThreads = (uint32_t)std::thread::hardware_concurrency();

  using IDBuilder = sycl::detail::Builder;

public:
  libCMBatch(KernelType Kernel) : MKernel(Kernel), spaceDim{1, 1, 1} {}

  // ID
  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<__Dims__>>::value>::type
  runIterationSpace(const sycl::range<__Dims__> Range) {
    sycl::detail::NDLoop<__Dims__>::iterate(
        Range, [&](const sycl::id<__Dims__> &ID) { argVector.push_back(ID); });
    for (int I = 0; I < __Dims__; ++I) {
      spaceDim[I] = (uint32_t)Range[I];
    }
    run();
  }

  // Item w/o offset
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, sycl::item<__Dims__, /*Offset=*/false>>::value>::type
  runIterationSpace(const sycl::range<__Dims__> Range) {
    sycl::detail::NDLoop<__Dims__>::iterate(
        Range, [&](const sycl::id<__Dims__> ID) {
          sycl::item<__Dims__, /*Offset=*/false> Item =
              IDBuilder::createItem<__Dims__, false>(Range, ID);
          argVector.push_back(Item);
        });
    for (int I = 0; I < __Dims__; ++I) {
      spaceDim[I] = (uint32_t)Range[I];
    }
    run();
  }

  // Item w/ offset
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, sycl::item<__Dims__, /*Offset=*/true>>::value>::type
  runIterationSpace(const sycl::range<__Dims__> Range,
                    const sycl::id<__Dims__> Offset) {
    sycl::detail::NDLoop<__Dims__>::iterate(
        Range, [&](const sycl::id<__Dims__> &ID) {
          sycl::id<__Dims__> OffsetID = ID + Offset;
          sycl::item<__Dims__, /*Offset=*/true> Item =
              IDBuilder::createItem<__Dims__, true>(Range, OffsetID, Offset);
          argVector.push_back(Item);
        });
    for (int I = 0; I < __Dims__; ++I) {
      spaceDim[I] = (uint32_t)Range[I];
    }
    run();
  }

  // NDItem
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, sycl::nd_item<__Dims__>>::value>::type
  runIterationSpace(const sycl::id<__Dims__> &GroupID,
                    const sycl::group<__Dims__> Group,
                    const sycl::range<__Dims__> LocalSize,
                    const sycl::range<__Dims__> GlobalSize,
                    const sycl::id<__Dims__> GlobalOffset) {
    sycl::detail::NDLoop<__Dims__>::iterate(
        LocalSize, [&](const sycl::id<__Dims__> &LocalID) {
          sycl::id<__Dims__> GlobalID =
              GroupID * LocalSize + LocalID + GlobalOffset;
          const sycl::item<__Dims__, /*Offset=*/true> GlobalItem =
              IDBuilder::createItem<__Dims__, true>(GlobalSize, GlobalID,
                                                    GlobalOffset);
          const sycl::item<__Dims__, /*Offset=*/false> LocalItem =
              IDBuilder::createItem<__Dims__, false>(LocalSize, LocalID);
          const sycl::nd_item<__Dims__> NDItem =
              IDBuilder::createNDItem<__Dims__>(GlobalItem, LocalItem, Group);
          argVector.push_back(NDItem);
        });
    for (int I = 0; I < __Dims__; ++I) {
      spaceDim[I] = (uint32_t)LocalSize[I];
    }
    run();
  }

private:
  /*
  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<__Dims__>>::value>::type
  run()
  {
    throw cl::sycl::feature_not_supported();
  }
  */

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<1>>::value>::type run() {
    struct LambdaWrapper_ID_1DIM *wrappedLambda_ID_1DIM =
        makeWrapper_ID_1DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ID_1DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ID_1DIM),
                      wrappedLambda_ID_1DIM);

    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<2>>::value>::type run() {
    struct LambdaWrapper_ID_2DIM *wrappedLambda_ID_2DIM =
        makeWrapper_ID_2DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ID_2DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ID_2DIM),
                      wrappedLambda_ID_2DIM);

    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::id<3>>::value>::type run() {
    struct LambdaWrapper_ID_3DIM *wrappedLambda_ID_3DIM =
        makeWrapper_ID_3DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ID_3DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ID_3DIM),
                      wrappedLambda_ID_3DIM);

    argVector.clear();
  }

  /*
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, sycl::item<__Dims__, false>>::value>::type
  run()
  {
    throw cl::sycl::feature_not_supported();
  }
  */

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::item<1, false>>::value>::type
  run() {
    struct LambdaWrapper_ITEM_1DIM *wrappedLambda_ITEM_1DIM =
        makeWrapper_ITEM_1DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ITEM_1DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_1DIM),
                      wrappedLambda_ITEM_1DIM);

    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::item<2, false>>::value>::type
  run() {
    struct LambdaWrapper_ITEM_2DIM *wrappedLambda_ITEM_2DIM =
        makeWrapper_ITEM_2DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ITEM_2DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_2DIM),
                      wrappedLambda_ITEM_2DIM);

    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::item<3, false>>::value>::type
  run() {
    struct LambdaWrapper_ITEM_3DIM *wrappedLambda_ITEM_3DIM =
        makeWrapper_ITEM_3DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ITEM_3DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_3DIM),
                      wrappedLambda_ITEM_3DIM);

    argVector.clear();
  }

  /*
  template <class ArgT = KernelArgType>
  typename std::enable_if<
      std::is_same<ArgT, sycl::item<__Dims__, true>>::value>::type
  run()
  {
    throw cl::sycl::feature_not_supported();
  }
  */

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::item<1, true>>::value>::type
  run() {
    struct LambdaWrapper_ITEM_OFFSET_1DIM *wrappedLambda_ITEM_OFFSET_1DIM =
        makeWrapper_ITEM_OFFSET_1DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ITEM_OFFSET_1DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_OFFSET_1DIM),
                      wrappedLambda_ITEM_OFFSET_1DIM);
    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::item<2, true>>::value>::type
  run() {
    struct LambdaWrapper_ITEM_OFFSET_2DIM *wrappedLambda_ITEM_OFFSET_2DIM =
        makeWrapper_ITEM_OFFSET_2DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ITEM_OFFSET_2DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_OFFSET_2DIM),
                      wrappedLambda_ITEM_OFFSET_2DIM);
    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::item<3, true>>::value>::type
  run() {
    struct LambdaWrapper_ITEM_OFFSET_3DIM *wrappedLambda_ITEM_OFFSET_3DIM =
        makeWrapper_ITEM_OFFSET_3DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_ITEM_OFFSET_3DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_ITEM_OFFSET_3DIM),
                      wrappedLambda_ITEM_OFFSET_3DIM);
    argVector.clear();
  }

  /*
  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT,
  sycl::nd_item<__Dims__>>::value>::type run()
  {
    throw cl::sycl::feature_not_supported();
  }
  */

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::nd_item<1>>::value>::type
  run() {
    struct LambdaWrapper_NDITEM_1DIM *wrappedLambda_NDITEM_1DIM =
        makeWrapper_NDITEM_1DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_NDITEM_1DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_NDITEM_1DIM),
                      wrappedLambda_NDITEM_1DIM);
    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::nd_item<2>>::value>::type
  run() {
    struct LambdaWrapper_NDITEM_2DIM *wrappedLambda_NDITEM_2DIM =
        makeWrapper_NDITEM_2DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_NDITEM_2DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_NDITEM_2DIM),
                      wrappedLambda_NDITEM_2DIM);
    argVector.clear();
  }

  template <class ArgT = KernelArgType>
  typename std::enable_if<std::is_same<ArgT, sycl::nd_item<3>>::value>::type
  run() {
    struct LambdaWrapper_NDITEM_3DIM *wrappedLambda_NDITEM_3DIM =
        makeWrapper_NDITEM_3DIM(argVector, MKernel);

    ESimdCPUKernel eSimdCPU((fptrVoid)invokeLambda_NDITEM_3DIM, spaceDim);

    eSimdCPU.launchMT(sizeof(struct LambdaWrapper_NDITEM_3DIM),
                      wrappedLambda_NDITEM_3DIM);
    argVector.clear();
  }
};

inline void __cm_mt_barrier() { cm_support::mt_barrier(); }

inline void __cm_set_slm_size(size_t size) { cm_support::set_slm_size(size); }

inline size_t __cm_get_slm_size() { return cm_support::get_slm_size(); }

inline char *__cm_get_slm() { return cm_support::get_slm(); }

#endif // __SYCL_DEVICE_ONLY__

#endif // _ESIMD_LIBCM_HPP_
