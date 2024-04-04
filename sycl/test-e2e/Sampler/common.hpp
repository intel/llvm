#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

using namespace sycl;

template <typename vecType, int numOfElems>
std::string vec2string(const vec<vecType, numOfElems> &vec) {
  std::string str = "";
  for (size_t i = 0; i < numOfElems - 1; ++i) {
    str += std::to_string(vec[i]) + ",";
  }
  str = "{" + str + std::to_string(vec[numOfElems - 1]) + "}";
  return str;
}

template <typename vecType, int numOfElems>
void check_pixel(const vec<vecType, numOfElems> &result,
                 const vec<vecType, numOfElems> &ref, int index) {
  // 0 ULP difference is allowed for integer type
  int precision = 0;
  // 1 ULP difference is allowed for float type
  if (std::is_floating_point<vecType>::value) {
    precision = 1;
  }

  int4 resultInt = result.template as<int4>();
  int4 refInt = ref.template as<int4>();
  int4 diff = resultInt - refInt;
  int isCorrect = all((diff <= precision) && (diff >= (-precision)));
  if (isCorrect) {
    std::cout << index << " -- " << vec2string(result) << std::endl;
  } else {
    std::string errMsg = "unexpected result: " + vec2string(result) +
                         " vs reference: " + vec2string(ref);
    std::cout << index << " -- " << errMsg << std::endl;
    exit(1);
  }
}

template <typename accType, typename pixelType>
void check_pixels(accType &pixels, const std::vector<pixelType> &ref,
                  size_t &offset) {
  for (int i = offset, ref_i = 0; i < ref.size(); i++, ref_i++) {
    pixelType testPixel = pixels[i];
    check_pixel(testPixel, ref[ref_i], i);
  }
  offset += ref.size();
}
