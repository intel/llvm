#ifndef MATH_UTILS
#include <complex>
#include <limits>
using namespace std;
// T must be float-point type
template <typename T>
bool is_about_FP(T x, T y) {
  if (x == y)
    return true;
  else {
    if (x != 0 && y != 0) {
      T max_v = fmax(abs(x), abs(y));
      return (abs(x - y) / max_v) <
              numeric_limits<T>::epsilon() * 100;
    }
    else {
      if (x != 0)
        return abs(x) < numeric_limits<T>::epsilon() * 100;
      else
        return abs(y) < numeric_limits<T>::epsilon() * 100;
    }
  }
}

#endif
