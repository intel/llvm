// Minimal benchmark harness for the SYCL PerformanceTests. Header-only, uses
// only the C++17 standard library. See README.md in this directory.
//
// Usage:
//   int main(int argc, char **argv) {
//     perf::add_case("my case", "ns", [&](perf::State &s) {
//       s.items(100);
//       s.measure([&] { for (int i = 0; i < 100; ++i) work(); });
//     });
//     return perf::main(argc, argv);
//   }
//
// The case function is invoked once per warm-up and once per timed
// repetition. Each invocation must call exactly one of s.measure() or
// s.manual(). For every case that succeeds one line is printed:
//   PERF_RESULT {"name":...,"variant":...,"unit":...,"value":<median>,
//                "stddev":...,"min":...,"reps":...,"lower_is_better":...}

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace perf {

struct Options {
  int warmup = 1;
  int reps = 10;
  bool lower_is_better = true;
};

// Thrown by check(); reported as a failure of the current case.
struct CheckFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void check(bool Cond, const char *Msg) {
  if (!Cond)
    throw CheckFailure(std::string("correctness check failed: ") + Msg);
}

inline void check(bool Cond, const std::string &Msg) {
  check(Cond, Msg.c_str());
}

struct Case;

class State {
public:
  // Times one call of F with std::chrono::steady_clock and records it in the
  // case's unit (which must be one of s, ms, us, ns).
  template <typename F> void measure(F &&Func) {
    if (NsPerUnit == 0)
      throw std::logic_error("perf::State::measure: unit must be s, ms, us or "
                             "ns; use manual() for other units");
    auto Start = std::chrono::steady_clock::now();
    std::forward<F>(Func)();
    auto End = std::chrono::steady_clock::now();
    record(std::chrono::duration<double, std::nano>(End - Start).count() /
           NsPerUnit);
  }

  // Records a value measured by the test itself, in the case's unit.
  void manual(double Value) { record(Value); }

  // The recorded value of this repetition is divided by K.
  void items(double K) {
    if (!(K > 0))
      throw std::invalid_argument("perf::State::items: K must be positive");
    Items = K;
  }

  bool is_warmup() const { return Warmup; }
  // 0-based index of this invocation, counting warm-up invocations too.
  int iteration() const { return Iteration; }

private:
  friend int run_case(const Case &, int, int, const std::string &);

  void record(double Value) {
    if (Recorded)
      throw std::logic_error(
          "perf::State: measure()/manual() called more than once per "
          "repetition");
    Recorded = true;
    Value /= Items;
    if (!std::isfinite(Value))
      throw std::runtime_error("perf::State: non-finite measurement");
    if (!Warmup)
      Values.push_back(Value);
  }

  double NsPerUnit = 0;
  double Items = 1;
  bool Warmup = false;
  bool Recorded = false;
  int Iteration = 0;
  std::vector<double> Values;
};

struct Case {
  std::string Name;
  std::string Unit;
  std::function<void(State &)> Fn;
  Options Opts;
};

namespace detail {

inline std::vector<Case> &cases() {
  static std::vector<Case> C;
  return C;
}

inline std::map<std::string, std::string> &args() {
  static std::map<std::string, std::string> A;
  return A;
}

[[noreturn]] inline void fatal(const std::string &Msg) {
  std::cerr << "perf_harness: error: " << Msg << std::endl;
  std::exit(2);
}

// Keeps names safe to embed in JSON without escaping.
inline bool valid_label(std::string_view S) {
  constexpr std::string_view Extra = " _.,:;<>()=+/*&#-";
  if (S.size() > 120)
    return false;
  for (char C : S)
    if (!((C >= 'A' && C <= 'Z') || (C >= 'a' && C <= 'z') ||
          (C >= '0' && C <= '9') || Extra.find(C) != std::string_view::npos))
      return false;
  return true;
}

inline void validate_label(std::string_view What, std::string_view S) {
  if (S.empty() && What == "case name")
    fatal("empty case name");
  if (!valid_label(S))
    fatal(std::string(What) + " \"" + std::string(S) +
          "\" is longer than 120 characters or contains characters outside "
          "[A-Za-z0-9 _.,:;<>()=+/*&#-]");
}

inline double ns_per_unit(const std::string &Unit) {
  if (Unit == "s")
    return 1e9;
  if (Unit == "ms")
    return 1e6;
  if (Unit == "us")
    return 1e3;
  if (Unit == "ns")
    return 1;
  return 0;
}

inline std::string num(double V) {
  char Buf[64];
  std::snprintf(Buf, sizeof(Buf), "%.9g", V);
  return Buf;
}

inline std::optional<int> parse_int(const std::string &S, int Min) {
  try {
    size_t Pos = 0;
    int V = std::stoi(S, &Pos);
    if (Pos == S.size() && V >= Min)
      return V;
  } catch (...) {
  }
  return std::nullopt;
}

} // namespace detail

inline void add_case(std::string Name, std::string Unit,
                     std::function<void(State &)> Fn, Options Opts = {}) {
  detail::validate_label("case name", Name);
  detail::validate_label("unit", Unit);
  for (const Case &C : detail::cases())
    if (C.Name == Name)
      detail::fatal("duplicate case name \"" + Name + "\"");
  detail::cases().push_back(
      Case{std::move(Name), std::move(Unit), std::move(Fn), Opts});
}

// Value of a --Key=value command-line argument. Only valid once perf::main()
// has parsed the command line, i.e. inside case functions.
inline std::string arg(const std::string &Key) {
  auto It = detail::args().find(Key);
  if (It == detail::args().end())
    throw std::runtime_error("missing required argument --" + Key + "=...");
  return It->second;
}

inline std::string arg(const std::string &Key, const std::string &Default) {
  auto It = detail::args().find(Key);
  return It == detail::args().end() ? Default : It->second;
}

// Returns 0 on success, 1 if the case threw (including failed checks).
inline int run_case(const Case &C, int Warmup, int Reps,
                    const std::string &Variant) {
  State S;
  S.NsPerUnit = detail::ns_per_unit(C.Unit);
  try {
    for (int I = 0; I < Warmup + Reps; ++I) {
      S.Warmup = I < Warmup;
      S.Iteration = I;
      S.Items = 1;
      S.Recorded = false;
      C.Fn(S);
      if (!S.Recorded)
        throw std::logic_error("case function called neither measure() nor "
                               "manual()");
    }
  } catch (const std::exception &E) {
    std::cerr << "perf_harness: case \"" << C.Name << "\" failed: " << E.what()
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "perf_harness: case \"" << C.Name
              << "\" failed: unknown exception" << std::endl;
    return 1;
  }

  std::vector<double> V = S.Values;
  std::sort(V.begin(), V.end());
  size_t N = V.size();
  double Median = N % 2 ? V[N / 2] : (V[N / 2 - 1] + V[N / 2]) / 2;
  double Mean = 0;
  for (double X : V)
    Mean += X;
  Mean /= N;
  double Var = 0;
  for (double X : V)
    Var += (X - Mean) * (X - Mean);
  double Stddev = N > 1 ? std::sqrt(Var / (N - 1)) : 0;

  std::cout << "PERF_RESULT {\"name\":\"" << C.Name << "\",\"variant\":\""
            << Variant << "\",\"unit\":\"" << C.Unit
            << "\",\"value\":" << detail::num(Median)
            << ",\"stddev\":" << detail::num(Stddev)
            << ",\"min\":" << detail::num(V.front()) << ",\"reps\":" << N
            << ",\"lower_is_better\":"
            << (C.Opts.lower_is_better ? "true" : "false") << "}" << std::endl;
  return 0;
}

// Parses the command line and runs the selected cases. Returns the process
// exit code: 0 if every selected case succeeded, nonzero otherwise.
inline int main(int argc, char **argv) {
  bool List = false;
  std::vector<std::string> Selected;
  std::optional<int> Reps, Warmup;
  std::string Variant;
  for (int I = 1; I < argc; ++I) {
    std::string A = argv[I];
    if (A == "--list") {
      List = true;
      continue;
    }
    size_t Eq = A.find('=');
    if (A.rfind("--", 0) != 0 || Eq == std::string::npos || Eq == 2)
      detail::fatal("unexpected argument \"" + A +
                    "\"; expected --list or --key=value");
    std::string Key = A.substr(2, Eq - 2), Value = A.substr(Eq + 1);
    if (Key == "case") {
      Selected.push_back(Value);
    } else if (Key == "reps") {
      if (!(Reps = detail::parse_int(Value, 1)))
        detail::fatal("--reps must be an integer >= 1");
    } else if (Key == "warmup") {
      if (!(Warmup = detail::parse_int(Value, 0)))
        detail::fatal("--warmup must be an integer >= 0");
    } else if (Key == "variant") {
      detail::validate_label("variant", Value);
      Variant = Value;
    } else {
      detail::args()[Key] = Value;
    }
  }

  for (const std::string &Name : Selected)
    if (std::none_of(detail::cases().begin(), detail::cases().end(),
                     [&](const Case &C) { return C.Name == Name; }))
      detail::fatal("no case named \"" + Name + "\" (see --list)");

  if (List) {
    for (const Case &C : detail::cases())
      std::cout << C.Name << "\n";
    return 0;
  }

  int Failures = 0;
  for (const Case &C : detail::cases()) {
    if (!Selected.empty() &&
        std::find(Selected.begin(), Selected.end(), C.Name) == Selected.end())
      continue;
    Failures += run_case(C, Warmup.value_or(C.Opts.warmup),
                         Reps.value_or(C.Opts.reps), Variant);
  }
  if (Failures)
    std::cerr << "perf_harness: " << Failures << " case(s) failed" << std::endl;
  return Failures ? 1 : 0;
}

} // namespace perf
