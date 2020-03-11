#ifndef MATH_UTILS
#include <complex>
#include <limits>
// T must be float-point type
template <typename T>
bool is_about_FP(T x, T y) {
  // At least one input is nan
  if (std::isnan(x) || std::isnan(y))
    return std::isnan(x) && std::isnan(y);

  // At least one input is inf
  else if (std::isinf(x) || std::isinf(y))
    return (x == y);

  // two finite
  else {
    if (x != 0 && y != 0) {
      T max_v = std::fmax(std::abs(x), std::abs(y));
      return (std::abs(x - y) / max_v) <
             std::numeric_limits<T>::epsilon() * 100;
    } else {
      if (x != 0)
        return std::abs(x) < std::numeric_limits<T>::epsilon() * 100;
      else
        return std::abs(y) < std::numeric_limits<T>::epsilon() * 100;
    }
  }
}

#endif
