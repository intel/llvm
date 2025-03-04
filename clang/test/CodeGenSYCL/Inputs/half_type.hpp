#define __SYCL_CONSTEXPR_HALF constexpr
using StorageT = _Float16;

class half {
public:
  half() = default;
  constexpr half(const half &) = default;
  constexpr half(half &&) = default;

  __SYCL_CONSTEXPR_HALF half(const float &rhs) : Data(rhs) {}

  constexpr half &operator=(const half &rhs) = default;

  __SYCL_CONSTEXPR_HALF half &operator/=(const half &rhs) {
    Data /= rhs.Data;
    return *this;
  }

#define OP(op, op_eq)                                                          \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend double operator op(const half lhs,              \
                                                  const double rhs) {          \
    double rtn = lhs;                                                          \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend double operator op(const double lhs,            \
                                                  const half rhs) {            \
    double rtn = lhs;                                                          \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend float operator op(const half lhs,               \
                                                 const float rhs) {            \
    float rtn = lhs;                                                           \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend float operator op(const float lhs,              \
                                                 const half rhs) {             \
    float rtn = lhs;                                                           \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const int rhs) {               \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const int lhs,                 \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const long rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const long lhs,                \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half lhs,                \
                                                const long long rhs) {         \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const long long lhs,           \
                                                const half rhs) {              \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half &lhs,               \
                                                const unsigned int &rhs) {     \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const unsigned int &lhs,       \
                                                const half &rhs) {             \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const half &lhs,               \
                                                const unsigned long &rhs) {    \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const unsigned long &lhs,      \
                                                const half &rhs) {             \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(                               \
      const half &lhs, const unsigned long long &rhs) {                        \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }                                                                            \
  __SYCL_CONSTEXPR_HALF friend half operator op(const unsigned long long &lhs, \
                                                const half &rhs) {             \
    half rtn = lhs;                                                            \
    rtn op_eq rhs;                                                             \
    return rtn;                                                                \
  }
  OP(/, /=)

#undef OP

  // Operator float
  __SYCL_CONSTEXPR_HALF operator float() const {
    return static_cast<float>(Data);
  }

private:
  __SYCL_CONSTEXPR_HALF StorageT getFPRep() const { return Data; }

  StorageT Data;
};

