#ifndef MATH_UTILS
#include <complex>
#include <limits>

// Since it is not proper to compare float point using operator ==, this
// function measures whether the result of cmath function from kernel is
// close to the reference and machine epsilon is used as threshold in this
// function. T must be float-point type.
template <typename T>
bool is_about_FP(T x, T y) {
  bool ret;

  // At least one input is nan
  if (std::isnan(x) || std::isnan(y))
    return std::isnan(x) && std::isnan(y);

  // At least one input is inf
  if (std::isinf(x) || std::isinf(y))
    ret = (x == y);

  // two finite
  else {
    T threshold = std::numeric_limits<T>::epsilon() * 100;
    if (x != 0 && y != 0) {
      T max_v = std::fmax(std::abs(x), std::abs(y));
      return std::abs(x - y) < threshold * max_v;
    } else {
      if (x != 0)
        ret = std::abs(x) < threshold;
      else
        ret = std::abs(y) < threshold;
    }
  }
  return ret;
}

#endif
