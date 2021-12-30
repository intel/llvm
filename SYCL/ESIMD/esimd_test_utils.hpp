//==--------- esimd_test_utils.hpp - DPC++ ESIMD on-device test utilities --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#define NOMINMAX

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace cl::sycl;

namespace esimd_test {

// This is the class provided to SYCL runtime by the application to decide
// on which device to run, or whether to run at all.
// When selecting a device, SYCL runtime first takes (1) a selector provided by
// the program or a default one and (2) the set of all available devices. Then
// it passes each device to the '()' operator of the selector. Device, for
// which '()' returned the highest number, is selected. If a negative number
// was returned for all devices, then the selection process will cause an
// exception.
class ESIMDSelector : public device_selector {
  // Require GPU device unless HOST is requested in SYCL_DEVICE_FILTER env
  virtual int operator()(const device &device) const {
    if (const char *dev_filter = getenv("SYCL_DEVICE_FILTER")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      if (filter_string.find("host") != std::string::npos)
        return device.is_host() ? 1000 : -1;
      std::cerr
          << "Supported 'SYCL_DEVICE_FILTER' env var values are 'gpu' and "
             "'host', '"
          << filter_string << "' does not contain such substrings.\n";
      return -1;
    }
    // If "SYCL_DEVICE_FILTER" not defined, only allow gpu device
    return device.is_gpu() ? 1000 : -1;
  }
};

inline auto createExceptionHandler() {
  return [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << "sycl::exception: " << e0.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "std::exception: " << e.what() << std::endl;
      } catch (...) {
        std::cout << "generic exception\n";
      }
    }
  };
}

template <typename T>
std::vector<T> read_binary_file(const char *fname, size_t num = 0) {
  std::vector<T> vec;
  std::ifstream ifs(fname, std::ios::in | std::ios::binary);
  if (ifs.good()) {
    ifs.unsetf(std::ios::skipws);
    std::streampos file_size;
    ifs.seekg(0, std::ios::end);
    file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    size_t max_num = file_size / sizeof(T);
    vec.resize(num ? (std::min)(max_num, num) : max_num);
    ifs.read(reinterpret_cast<char *>(vec.data()), vec.size() * sizeof(T));
  }
  return vec;
}

template <typename T>
bool write_binary_file(const char *fname, const std::vector<T> &vec,
                       size_t num = 0) {
  std::ofstream ofs(fname, std::ios::out | std::ios::binary);
  if (ofs.good()) {
    ofs.write(reinterpret_cast<const char *>(&vec[0]),
              (num ? num : vec.size()) * sizeof(T));
    ofs.close();
  }
  return !ofs.bad();
}

template <typename T>
bool cmp_binary_files(const char *fname1, const char *fname2, T tolerance) {
  const auto vec1 = read_binary_file<T>(fname1);
  const auto vec2 = read_binary_file<T>(fname2);
  if (vec1.size() != vec2.size()) {
    std::cerr << fname1 << " size is " << vec1.size();
    std::cerr << " whereas " << fname2 << " size is " << vec2.size()
              << std::endl;
    return false;
  }
  for (size_t i = 0; i < vec1.size(); i++) {
    if (abs(vec1[i] - vec2[i]) > tolerance) {
      std::cerr << "Mismatch at " << i << ' ';
      if (sizeof(T) == 1) {
        std::cerr << (int)vec1[i] << " vs " << (int)vec2[i] << std::endl;
      } else {
        std::cerr << vec1[i] << " vs " << vec2[i] << std::endl;
      }
      return false;
    }
  }
  return true;
}

// dump every element of sequence [first, last) to std::cout
template <typename ForwardIt> void dump_seq(ForwardIt first, ForwardIt last) {
  using ValueT = typename std::iterator_traits<ForwardIt>::value_type;
  std::copy(first, last, std::ostream_iterator<ValueT>{std::cout, " "});
  std::cout << std::endl;
}

// Checks wether ranges [first, last) and [ref_first, ref_last) are equal.
// If a mismatch is found, dumps elements that differ and returns true,
// otherwise false is returned.
template <typename ForwardIt, typename RefForwardIt, typename BinaryPredicateT>
bool check_fail_seq(ForwardIt first, ForwardIt last, RefForwardIt ref_first,
                    RefForwardIt ref_last, BinaryPredicateT is_equal) {
  auto mism = std::mismatch(first, last, ref_first, is_equal);
  if (mism.first != last) {
    std::cout << "mismatch: returned " << *mism.first << std::endl;
    std::cout << "          expected " << *mism.second << std::endl;
    return true;
  }
  return false;
}

template <typename ForwardIt, typename RefForwardIt>
bool check_fail_seq(ForwardIt first, ForwardIt last, RefForwardIt ref_first,
                    RefForwardIt ref_last) {
  return check_fail_seq(
      first, last, ref_first, ref_last,
      [](const auto &lhs, const auto &rhs) { return lhs == rhs; });
}

// analog to C++20 bit_cast
template <typename To, typename From,
          typename std::enable_if<(sizeof(To) == sizeof(From)) &&
                                      std::is_trivially_copyable<From>::value &&
                                      std::is_trivial<To>::value,
                                  int>::type = 0>
To bit_cast(const From &src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

// Timer class for measuring elasped time
class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

// e0 is the first event, en is the last event
// find the time difference between the starting time of the e0 and
// the ending time of en, return micro-second
inline double report_time(const std::string &msg, event e0, event en) {
  cl_ulong time_start =
      e0.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      en.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  // cerr << msg << elapsed << " msecs" << std::endl;
  return elapsed;
}

void display_timing_stats(double const kernelTime,
                          unsigned int const uiNumberOfIterations,
                          double const overallTime) {
  std::cout << "Number of iterations: " << uiNumberOfIterations << "\n";
  std::cout << "[KernelTime]:" << kernelTime << "\n";
  std::cout << "[OverallTime][Primary]:" << overallTime << "\n";
}

// Get signed integer of given byte size.
template <int N>
using int_type_t = std::conditional_t<
    N == 1, int8_t,
    std::conditional_t<
        N == 2, int16_t,
        std::conditional_t<N == 4, int32_t,
                           std::conditional_t<N == 8, int64_t, void>>>>;

enum class BinaryOp {
  add,
  sub,
  mul,
  div, // <-- only this and above are supported for floating point types
  rem,
  shl,
  shr,
  bit_or,
  bit_and,
  bit_xor, // <-- only this and above are supported for integer types
  log_or,  // only for simd_mask
  log_and  // only for simd_mask
};

enum class UnaryOp {
  minus_minus_pref,
  minus_minus_inf,
  plus_plus_pref,
  plus_plus_inf,
  minus,
  plus,
  bit_not,
  log_not
};

enum class CmpOp { lt, lte, eq, ne, gte, gt };

#define __bin_case(x, s)                                                       \
  case BinaryOp::x:                                                            \
    return s
#define __cmp_case(x, s)                                                       \
  case CmpOp::x:                                                               \
    return s
#define __un_case(x, s)                                                        \
  case UnaryOp::x:                                                             \
    return s

template <class OpClass> const char *Op2Str(OpClass op) {
  if constexpr (std::is_same_v<OpClass, BinaryOp>) {
    switch (op) {
      __bin_case(add, "+");
      __bin_case(sub, "-");
      __bin_case(mul, "*");
      __bin_case(div, "/");
      __bin_case(rem, "%");
      __bin_case(shl, "<<");
      __bin_case(shr, ">>");
      __bin_case(bit_or, "|");
      __bin_case(bit_and, "&");
      __bin_case(bit_xor, "^");
      __bin_case(log_or, "||");
      __bin_case(log_and, "&&");
    }
  } else if constexpr (std::is_same_v<OpClass, CmpOp>) {
    switch (op) {
      __cmp_case(lt, "<");
      __cmp_case(lte, "<=");
      __cmp_case(eq, "==");
      __cmp_case(ne, "!=");
      __cmp_case(gte, ">=");
      __cmp_case(gt, ">");
    }
  } else if constexpr (std::is_same_v<OpClass, UnaryOp>) {
    switch (op) {
      __un_case(minus, "-x");
      __un_case(minus_minus_pref, "--x");
      __un_case(minus_minus_inf, "x--");
      __un_case(plus, "+x");
      __un_case(plus_plus_pref, "++x");
      __un_case(plus_plus_inf, "x++");
      __un_case(bit_not, "~x");
      __un_case(log_not, "!x");
    }
  }
}

template <class OpClass, OpClass Op, class T1, class T2>
inline auto binary_op(T1 x, T2 y) {
  if constexpr (std::is_same_v<OpClass, BinaryOp>) {
    if constexpr (Op == BinaryOp::add)
      return x + y;
    else if constexpr (Op == BinaryOp::sub)
      return x - y;
    else if constexpr (Op == BinaryOp::mul)
      return x * y;
    else if constexpr (Op == BinaryOp::div)
      return x / y;
    else if constexpr (Op == BinaryOp::rem)
      return x % y;
    else if constexpr (Op == BinaryOp::shl)
      return x << y;
    else if constexpr (Op == BinaryOp::shr)
      return x >> y;
    else if constexpr (Op == BinaryOp::bit_or)
      return x | y;
    else if constexpr (Op == BinaryOp::bit_and)
      return x & y;
    else if constexpr (Op == BinaryOp::bit_xor)
      return x ^ y;
    else if constexpr (Op == BinaryOp::log_or)
      return x || y;
    else if constexpr (Op == BinaryOp::log_and)
      return x && y;
  } else if constexpr (std::is_same_v<OpClass, CmpOp>) {
    if constexpr (Op == CmpOp::lt)
      return x < y;
    else if constexpr (Op == CmpOp::lte)
      return x <= y;
    else if constexpr (Op == CmpOp::eq)
      return x == y;
    else if constexpr (Op == CmpOp::ne)
      return x != y;
    else if constexpr (Op == CmpOp::gte)
      return x >= y;
    else if constexpr (Op == CmpOp::gt)
      return x > y;
  }
}

template <UnaryOp Op, class T> inline auto unary_op(T x) {
  if constexpr (Op == UnaryOp::minus)
    return -x;
  else if constexpr (Op == UnaryOp::minus_minus_pref) {
    --x;
    return x;
  } else if constexpr (Op == UnaryOp::minus_minus_inf) {
    x--;
    return x;
  } else if constexpr (Op == UnaryOp::plus)
    return +x;
  else if constexpr (Op == UnaryOp::plus_plus_pref) {
    ++x;
    return x;
  } else if constexpr (Op == UnaryOp::plus_plus_inf) {
    x++;
    return x;
  } else if constexpr (Op == UnaryOp::bit_not)
    return ~x;
  else if constexpr (Op == UnaryOp::log_not)
    return !x;
}

template <int From, int To> struct ConstexprForLoop {
  template <class Action> static inline void unroll(Action act) {
    if constexpr (From < To) {
      act.template run<From>();
    }
    if constexpr (From < To - 1) {
      ConstexprForLoop<From + 1, To>::unroll(act);
    }
  }
};

template <class OpClass, OpClass... Ops> struct OpSeq {
  static constexpr size_t size = sizeof...(Ops);
  template <int I> static constexpr OpClass get() {
    std::array<OpClass, size> arr = {Ops...};
    return arr[I];
  }
};

template <BinaryOp... Ops> using BinaryOpSeq = OpSeq<BinaryOp, Ops...>;

static constexpr BinaryOpSeq<BinaryOp::add, BinaryOp::sub, BinaryOp::mul,
                             BinaryOp::div>
    ArithBinaryOps{};

static constexpr BinaryOpSeq<BinaryOp::add, BinaryOp::sub, BinaryOp::mul,
                             BinaryOp::div, BinaryOp::rem, BinaryOp::shl,
                             BinaryOp::shr, BinaryOp::bit_or, BinaryOp::bit_and,
                             BinaryOp::bit_xor>
    IntBinaryOps{};

static constexpr BinaryOpSeq<BinaryOp::add, BinaryOp::sub, BinaryOp::mul,
                             BinaryOp::div, BinaryOp::rem, BinaryOp::bit_or,
                             BinaryOp::bit_and, BinaryOp::bit_xor>
    IntBinaryOpsNoShift{};

static constexpr OpSeq<CmpOp, CmpOp::lt, CmpOp::lte, CmpOp::eq, CmpOp::ne,
                       CmpOp::gte, CmpOp::gt>
    CmpOps{};

static constexpr OpSeq<UnaryOp, UnaryOp::minus, UnaryOp::minus_minus_pref,
                       UnaryOp::minus_minus_inf, UnaryOp::plus,
                       UnaryOp::plus_plus_pref, UnaryOp::plus_plus_inf,
                       UnaryOp::bit_not, UnaryOp::log_not>
    UnaryOps{};

// Binary operations iteration

template <class T1, class T2, class F, class OpClass, OpClass... Ops>
struct ApplyBinaryOpAction {
  T1 x;
  T2 y;
  F f;

  ApplyBinaryOpAction(T1 x, T2 y, F f) : x(x), y(y), f(f) {}

  template <int OpIndex> inline void run() {
    constexpr OpClass arr[] = {Ops...};
    constexpr auto op = arr[OpIndex];
    f(binary_op<OpClass, op>(x, y), op, OpIndex);
  }
};

template <class T1, class T2, class F, class OpClass, OpClass... Ops>
inline void apply_ops(OpSeq<OpClass, Ops...> ops, T1 x, T2 y, F f) {
  ApplyBinaryOpAction<T1, T2, F, OpClass, Ops...> act(x, y, f);
  ConstexprForLoop<0, sizeof...(Ops)>::unroll(act);
}

// Unary operations iteration

template <class T, class F, UnaryOp... Ops> struct ApplyUnaryOpAction {
  T x;
  F f;

  ApplyUnaryOpAction(T x, F f) : x(x), f(f) {}

  template <int OpIndex> inline void run() {
    constexpr UnaryOp arr[] = {Ops...};
    constexpr auto op = arr[OpIndex];
    f(unary_op<op>(x), op, OpIndex);
  }
};

template <class T, class F, UnaryOp... Ops>
inline void apply_unary_ops(OpSeq<UnaryOp, Ops...> ops, T x, F f) {
  ApplyUnaryOpAction<T, F, Ops...> act(x, f);
  ConstexprForLoop<0, sizeof...(Ops)>::unroll(act);
}

// All operations

template <class F, class OpClass, OpClass... Ops> struct IterateOpAction {
  F f;
  IterateOpAction(F f) : f(f) {}

  template <int OpIndex> inline void run() {
    constexpr OpClass arr[] = {Ops...};
    constexpr auto op = arr[OpIndex];
    f(op);
  }
};

template <class F, class OpClass, OpClass... Ops>
inline void iterate_ops(OpSeq<OpClass, Ops...> ops, F f) {
  IterateOpAction<F, OpClass, Ops...> act(f);
  ConstexprForLoop<0, sizeof...(Ops)>::unroll(act);
}

} // namespace esimd_test
