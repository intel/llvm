// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Temporarily disable test on Windows due to regressions in GPU driver.
// UNSUPPORTED: cuda || hip, windows

#include <sycl/sycl.hpp>

#include <iostream>

class test_1d_class;
class test_2d_class;
class test_3d_class;

namespace s = sycl;

template <typename dataT>
bool check_result(dataT resultData, dataT expectedData, float epsilon = 0.1) {
  bool correct = true;
  if (std::abs(resultData.r() - expectedData.r()) > epsilon)
    correct = false;
  if (std::abs(resultData.g() - expectedData.g()) > epsilon)
    correct = false;
  if (std::abs(resultData.b() - expectedData.b()) > epsilon)
    correct = false;
  if (std::abs(resultData.a() - expectedData.a()) > epsilon)
    correct = false;
#ifdef DEBUG_OUTPUT
  if (!correct) {
    std::cout << "Expected: " << expectedData.r() << ", " << expectedData.g()
              << ", " << expectedData.b() << ", " << expectedData.a() << "\n";
    std::cout << "Got:      " << resultData.r() << ", " << resultData.g()
              << ", " << resultData.b() << ", " << resultData.a() << "\n";
  }
#endif // DEBUG_OUTPUT
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test1d_coord(dataT *hostPtr, coordT coord, dataT expectedColour) {
  dataT resultData;

  s::sampler testSampler(s::coordinate_normalization_mode::unnormalized,
                         s::addressing_mode::clamp, s::filtering_mode::linear);

  s::default_selector selector;

  { // Scope everything to force destruction
    s::image<1> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<1>{3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    s::queue myQueue(selector);
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_1d_class>([=]() {
        dataT RetColor = imageAcc.read(coord, testSampler);
        resultDataAcc[0] = RetColor;
      });
    });
  }
  bool correct = check_result(resultData, expectedColour);
#ifdef DEBUG_OUTPUT
  if (!correct) {
    std::cout << "Coord: " << coord << "\n";
  }
#endif // DEBUG_OUTPUT
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test2d_coord(dataT *hostPtr, coordT coord, dataT expectedColour) {
  dataT resultData;

  s::sampler testSampler(s::coordinate_normalization_mode::unnormalized,
                         s::addressing_mode::clamp, s::filtering_mode::linear);

  s::default_selector selector;

  { // Scope everything to force destruction
    s::image<2> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<2>{3, 3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    s::queue myQueue(selector);
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_2d_class>([=]() {
        dataT RetColor = imageAcc.read(coord, testSampler);
        resultDataAcc[0] = RetColor;
      });
    });
  }
  bool correct = check_result(resultData, expectedColour);
#ifdef DEBUG_OUTPUT
  if (!correct) {
    std::cout << "Coord: " << coord.x() << ", " << coord.y() << "\n";
  }
#endif // DEBUG_OUTPUT
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test3d_coord(dataT *hostPtr, coordT coord, dataT expectedColour) {
  dataT resultData;

  s::sampler testSampler(s::coordinate_normalization_mode::unnormalized,
                         s::addressing_mode::clamp, s::filtering_mode::linear);

  s::default_selector selector;

  { // Scope everything to force destruction
    s::image<3> image(hostPtr, s::image_channel_order::rgba, channelType,
                      s::range<3>{3, 3, 3});

    s::buffer<dataT, 1> resultDataBuf(&resultData, s::range<1>(1));

    // Do the test by reading a single pixel from the image
    s::queue myQueue(selector);
    myQueue.submit([&](s::handler &cgh) {
      auto imageAcc = image.get_access<dataT, s::access::mode::read>(cgh);
      s::accessor<dataT, 1, s::access::mode::write> resultDataAcc(resultDataBuf,
                                                                  cgh);

      cgh.single_task<test_3d_class>([=]() {
        dataT RetColor = imageAcc.read(coord, testSampler);
        resultDataAcc[0] = RetColor;
      });
    });
  }
  bool correct = check_result(resultData, expectedColour);
#ifdef DEBUG_OUTPUT
  if (!correct) {
    std::cout << "Coord: " << coord.x() << ", " << coord.y() << ", "
              << coord.z() << "\n";
  }
#endif // DEBUG_OUTPUT
  return correct;
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test1d(coordT coord, dataT expectedResult) {
  dataT hostPtr[3];
  for (int i = 0; i < 3; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);
  return test1d_coord<dataT, coordT, channelType>(hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test2d(coordT coord, dataT expectedResult) {
  dataT hostPtr[9];
  for (int i = 0; i < 9; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);
  return test2d_coord<dataT, coordT, channelType>(hostPtr, coord,
                                                  expectedResult);
}

template <typename dataT, typename coordT, s::image_channel_type channelType>
bool test3d(coordT coord, dataT expectedResult) {
  dataT hostPtr[27];
  for (int i = 0; i < 27; i++)
    hostPtr[i] = dataT(0 + i, 20 + i, 40 + i, 60 + i);
  return test3d_coord<dataT, coordT, channelType>(hostPtr, coord,
                                                  expectedResult);
}

int main() {

  bool passed = true;

  // 1d image read tests
  if (!test1d<s::float4, float, s::image_channel_type::fp32>(
          0.0f, s::float4(0, 10, 20, 30)))
    passed = false;
  if (!test1d<s::float4, float, s::image_channel_type::fp32>(
          0.5f, s::float4(0, 20, 40, 60)))
    passed = false;
  if (!test1d<s::float4, float, s::image_channel_type::fp32>(
          0.9f, s::float4(0.4, 20.4, 40.4, 60.4)))
    passed = false;

  // 2d image read tests
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.0f, 0.0f), s::float4(0, 5, 10, 15)))
    passed = false;
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.5f, 0.0f), s::float4(0, 10, 20, 30)))
    passed = false;
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.0f, 0.5f), s::float4(0, 10, 20, 30)))
    passed = false;
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.5f, 0.5f), s::float4(0, 20, 40, 60)))
    passed = false;
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.9f, 0.0f), s::float4(0.2, 10.2, 20.2, 30.2)))
    passed = false;
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.0f, 0.9f), s::float4(0.6, 10.6, 20.6, 30.6)))
    passed = false;
  if (!test2d<s::float4, s::float2, s::image_channel_type::fp32>(
          s::float2(0.9f, 0.9f), s::float4(1.6, 21.6, 41.6, 61.6)))
    passed = false;

  // 3d image read tests
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.0f, 0.0f, 0.0f, 0.0f), s::float4(0, 2.5, 5, 7.5)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.5f, 0.0f, 0.0f, 0.0f), s::float4(0, 5, 10, 15)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.0f, 0.5f, 0.0f, 0.0f), s::float4(0, 5, 10, 15)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.0f, 0.0f, 0.5f, 0.0f), s::float4(0, 5, 10, 15)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.5f, 0.5f, 0.5f, 0.0f), s::float4(0, 20, 40, 60)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.9f, 0.0f, 0.0f, 0.0f), s::float4(0.1, 5.1, 10.1, 15.1)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.0f, 0.9f, 0.0f, 0.0f), s::float4(0.3, 5.3, 10.3, 15.3)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.0f, 0.0f, 0.9f, 0.0f), s::float4(0.9, 5.9, 10.9, 15.9)))
    passed = false;
  if (!test3d<s::float4, s::float4, s::image_channel_type::fp32>(
          s::float4(0.9f, 0.9f, 0.9f, 0.0f), s::float4(5.2, 25.2, 45.2, 65.2)))
    passed = false;

  return passed ? 0 : -1;
}
