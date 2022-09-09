// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out
//
//==-------image_constructors.cpp - SYCL image constructors basic test------==//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Tests the constructors, get_count and get_range APIs.

#include <cassert>
#include <sycl/sycl.hpp>

void no_delete(void *) {}

template <int Dims>
void test_constructors(sycl::range<Dims> r, void *imageHostPtr) {

  sycl::image_channel_order channelOrder = sycl::image_channel_order::rgbx;
  sycl::image_channel_type channelType =
      sycl::image_channel_type::unorm_short_565;
  unsigned int elementSize = 2; // 2 bytes
  int numElems = r.size();
  sycl::property_list propList{}; // empty property list

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const property_list& = {})
   */
  {
    sycl::image<Dims> img =
        sycl::image<Dims>(imageHostPtr, channelOrder, channelType, r);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&, const property_list&)
   */
  {
    sycl::image<Dims> img =
        sycl::image<Dims>(imageHostPtr, channelOrder, channelType, r, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&, allocator,
   *              const property_list& = {})
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img =
        sycl::image<Dims>(imageHostPtr, channelOrder, channelType, r, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&, allocator,
   *              const property_list&)
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img = sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }
  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const property_list& = {})
   */
  {
    const auto constHostPtr = imageHostPtr;
    sycl::image<Dims> img =
        sycl::image<Dims>(constHostPtr, channelOrder, channelType, r);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<Dims>&, const property_list&)
   */
  {
    const auto constHostPtr = imageHostPtr;
    sycl::image<Dims> img =
        sycl::image<Dims>(constHostPtr, channelOrder, channelType, r, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<Dims>&, allocator,
   *              const property_list& = {})
   */
  {
    const auto constHostPtr = imageHostPtr;
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img =
        sycl::image<Dims>(constHostPtr, channelOrder, channelType, r, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (const void*, image_channel_order,
   *              image_channel_type, const range<Dims>&, allocator,
   *              const property_list&)
   */
  {
    const auto constHostPtr = imageHostPtr;
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img = sycl::image<Dims>(
        constHostPtr, channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const property_list& = {})
   */
  {
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img =
        sycl::image<Dims>(hostPointer, channelOrder, channelType, r);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&, const property_list&)
   */
  {
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img =
        sycl::image<Dims>(hostPointer, channelOrder, channelType, r, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&, allocator,
   *              const property_list& = {})
   */
  {
    sycl::image_allocator imgAlloc;
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img =
        sycl::image<Dims>(hostPointer, channelOrder, channelType, r, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&, allocator,
   *              const property_list&)
   */
  {
    sycl::image_allocator imgAlloc;
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img = sycl::image<Dims>(
        hostPointer, channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, const property_list& = {})
   */
  {
    sycl::image<Dims> img = sycl::image<Dims>(channelOrder, channelType, r);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, const property_list&)
   */
  {
    sycl::image<Dims> img =
        sycl::image<Dims>(channelOrder, channelType, r, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, allocator, const property_list& = {})
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img =
        sycl::image<Dims>(channelOrder, channelType, r, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, allocator, const property_list&)
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img =
        sycl::image<Dims>(channelOrder, channelType, r, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }
}

template <int Dims>
void test_constructors_with_pitch(sycl::range<Dims> r,
                                  sycl::range<Dims - 1> pitch,
                                  void *imageHostPtr) {

  sycl::image_channel_order channelOrder = sycl::image_channel_order::rgbx;
  sycl::image_channel_type channelType =
      sycl::image_channel_type::unorm_short_565;
  unsigned int elementSize = 2; // 2 bytes for short_565
  int numElems = r.size();
  sycl::property_list propList{}; // empty property list

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, const property_list& = {})
   */
  {
    sycl::image<Dims> img =
        sycl::image<Dims>(imageHostPtr, channelOrder, channelType, r, pitch);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, const property_list&)
   */
  {
    sycl::image<Dims> img = sycl::image<Dims>(imageHostPtr, channelOrder,
                                              channelType, r, pitch, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, allocator,
   *              const property_list& = {})
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img = sycl::image<Dims>(imageHostPtr, channelOrder,
                                              channelType, r, pitch, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (void *, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, allocator, const property_list&)
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img = sycl::image<Dims>(
        imageHostPtr, channelOrder, channelType, r, pitch, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, const property_list& = {})
   */
  {
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img =
        sycl::image<Dims>(hostPointer, channelOrder, channelType, r, pitch);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, const property_list&)
   */
  {
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img = sycl::image<Dims>(hostPointer, channelOrder,
                                              channelType, r, pitch, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, allocator,
   *              const property_list& = {})
   */
  {
    sycl::image_allocator imgAlloc;
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img = sycl::image<Dims>(hostPointer, channelOrder,
                                              channelType, r, pitch, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (std::shared_ptr<void>&, image_channel_order,
   *              image_channel_type, const range<Dims>&,
   *              const range<Dims - 1>&, allocator, const property_list&)
   */
  {
    sycl::image_allocator imgAlloc;
    auto hostPointer = std::shared_ptr<void>(imageHostPtr, &no_delete);
    sycl::image<Dims> img = sycl::image<Dims>(
        hostPointer, channelOrder, channelType, r, pitch, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, const range<Dims - 1>&,
   *              const property_list& = {})
   */
  {
    sycl::image<Dims> img =
        sycl::image<Dims>(channelOrder, channelType, r, pitch);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, const range<Dims - 1>&,
   *              const property_list&)
   */
  {
    sycl::image<Dims> img =
        sycl::image<Dims>(channelOrder, channelType, r, pitch, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, const range<Dims - 1>&, allocator,
   *              const property_list& = {})
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img =
        sycl::image<Dims>(channelOrder, channelType, r, pitch, imgAlloc);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }

  /* Constructor (image_channel_order, image_channel_type,
   *              const range<Dims>&, const range<Dims - 1>&, allocator,
   *              const property_list&)
   */
  {
    sycl::image_allocator imgAlloc;
    sycl::image<Dims> img = sycl::image<Dims>(channelOrder, channelType, r,
                                              pitch, imgAlloc, propList);
    assert(img.get_count() == numElems);
    assert(img.get_range() == r);
  }
}

int main() {

  int imageHostPtr[48]; // 3*2*4*(2 bytes per element) = 48
  for (int i = 0; i < 48; i++)
    imageHostPtr[i] = i; // Maximum number of elements.

  // Ranges
  sycl::range<1> r1(3);
  sycl::range<2> r2(3, 2);
  sycl::range<3> r3(3, 2, 4);

  // Pitches
  sycl::range<1> pitch2(6);     // range is 3; elementSize = 2.
  sycl::range<2> pitch3(6, 12); // range is 3,2; elementSize = 2.

  // Constructors without Pitch
  test_constructors<1>(r1, imageHostPtr);
  test_constructors<2>(r2, imageHostPtr);
  test_constructors<3>(r3, imageHostPtr);

  // Constructors with Pitch
  test_constructors_with_pitch<2>(r2, pitch2, imageHostPtr);
  test_constructors_with_pitch<3>(r3, pitch3, imageHostPtr);

  return 0;
}
