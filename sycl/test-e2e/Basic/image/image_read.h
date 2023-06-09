/**
 * Tests clamp_to_edge addressing mode with nearest and linear filtering modes
 * on a 4x4 image.
 *
 * Expected addressing mode and filtering results are given by the algorithm in
 * the OpenCL 1.2 specification, Section 8. Image Addressing and Filtering
 *
 * https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf#page=329
 */

#include <sycl/sycl.hpp>

#include <iostream>

template <typename T> class test_1d_class;
template <typename T> class test_2d_class;
template <typename T> class test_3d_class;

namespace s = sycl;

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test1d_coord(s::queue myQueue, dataT *hostPtr, coordT coord,
                  dataT expectedColour) {
  dataT resultData;

  { // Scope everything to force destruction
    s::image<1> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<1>{3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_1d_class<dataT>>([=]() {
        dataT RetColor = imageAcc.read(coord);
        resultDataAcc[0] = RetColor;
      });
    });
  }
#ifdef DEBUG_OUTPUT
  std::cout << "Expected: " << expectedColour.r() << ", " << expectedColour.g()
            << ", " << expectedColour.b() << ", " << expectedColour.a() << "\n";
  std::cout << "Got:      " << resultData.r() << ", " << resultData.g() << ", "
            << resultData.b() << ", " << resultData.a() << "\n";
#endif // DEBUG_OUTPUT
  bool correct = true;
  if (resultData.r() != expectedColour.r())
    correct = false;
  if (resultData.g() != expectedColour.g())
    correct = false;
  if (resultData.b() != expectedColour.b())
    correct = false;
  if (resultData.a() != expectedColour.a())
    correct = false;
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test2d_coord(s::queue myQueue, dataT *hostPtr, coordT coord,
                  dataT expectedColour) {
  dataT resultData;

  { // Scope everything to force destruction
    s::image<2> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<2>{3, 3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_2d_class<dataT>>([=]() {
        dataT RetColor = imageAcc.read(coord);
        resultDataAcc[0] = RetColor;
      });
    });
  }

#ifdef DEBUG_OUTPUT
  std::cout << "Got:      " << resultData.r() << ", " << resultData.g();
#endif // DEBUG_OUTPUT

  bool correct = true;
  if (resultData.r() != expectedColour.r())
    correct = false;
  if (resultData.g() != expectedColour.g())
    correct = false;
  if (resultData.b() != expectedColour.b())
    correct = false;
  if (resultData.a() != expectedColour.a())
    correct = false;
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test3d_coord(s::queue myQueue, dataT *hostPtr, coordT coord,
                  dataT expectedColour) {
  dataT resultData;

  { // Scope everything to force destruction
    s::image<3> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<3>{3, 3, 3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_3d_class<dataT>>([=]() {
        dataT RetColor = imageAcc.read(coord);
        resultDataAcc[0] = RetColor;
      });
    });
  }
#ifdef DEBUG_OUTPUT
  std::cout << "Expected: " << expectedColour.r() << ", " << expectedColour.g()
            << ", " << expectedColour.b() << ", " << expectedColour.a() << "\n";
  std::cout << "Got:      " << resultData.r() << ", " << resultData.g() << ", "
            << resultData.b() << ", " << resultData.a() << "\n";
#endif // DEBUG_OUTPUT
  bool correct = true;
  if (resultData.r() != expectedColour.r())
    correct = false;
  if (resultData.g() != expectedColour.g())
    correct = false;
  if (resultData.b() != expectedColour.b())
    correct = false;
  if (resultData.a() != expectedColour.a())
    correct = false;
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test1d(s::queue myQueue, coordT coord, dataT expectedResult) {
  dataT hostPtr[3];
  for (int i = 0; i < 3; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);

  return test1d_coord<dataT, coordT, channelType>(myQueue, hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test2d(s::queue myQueue, coordT coord, dataT expectedResult) {
  dataT hostPtr[9];
  for (int i = 0; i < 9; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);

  return test2d_coord<dataT, coordT, channelType>(myQueue, hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test3d(s::queue myQueue, coordT coord, dataT expectedResult) {
  dataT hostPtr[27];
  for (int i = 0; i < 27; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);

  return test3d_coord<dataT, coordT, channelType>(myQueue, hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, s::image_channel_type channelType>
bool test(s::queue myQueue) {
  bool passed = true;
  // 1d image tests
  if (!test1d<dataT, int, channelType>(myQueue, 0, dataT(0, 20, 40, 60)))
    passed = false;

  if (!test1d<dataT, int, channelType>(myQueue, 1, dataT(1, 21, 41, 61)))
    passed = false;

  if (!test1d<dataT, int, channelType>(myQueue, 2, dataT(2, 22, 42, 62)))
    passed = false;

  // 2d image tests
  if (!test2d<dataT, s::int2, channelType>(myQueue, s::int2(0, 0),
                                           dataT(0, 20, 40, 60)))
    passed = false;

  if (!test2d<dataT, s::int2, channelType>(myQueue, s::int2(1, 0),
                                           dataT(1, 21, 41, 61)))
    passed = false;

  if (!test2d<dataT, s::int2, channelType>(myQueue, s::int2(0, 1),
                                           dataT(3, 23, 43, 63)))
    passed = false;

  if (!test2d<dataT, s::int2, channelType>(myQueue, s::int2(1, 1),
                                           dataT(4, 24, 44, 64)))
    passed = false;

  // 3d image tests
  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(0, 0, 0, 0),
                                           dataT(0, 20, 40, 60)))
    passed = false;

  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(1, 0, 0, 0),
                                           dataT(1, 21, 41, 61)))
    passed = false;

  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(0, 1, 0, 0),
                                           dataT(3, 23, 43, 63)))
    passed = false;

  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(1, 1, 0, 0),
                                           dataT(4, 24, 44, 64)))
    passed = false;

  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(1, 0, 1, 0),
                                           dataT(10, 30, 50, 70)))
    passed = false;

  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(0, 1, 1, 0),
                                           dataT(12, 32, 52, 72)))
    passed = false;

  if (!test3d<dataT, s::int4, channelType>(myQueue, s::int4(1, 1, 1, 0),
                                           dataT(13, 33, 53, 73)))
    passed = false;

  return passed;
}