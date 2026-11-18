// Minimal benchmark harness for the SYCL PerformanceTests. Header-only, uses
// only the C++17 standard library. See README.md in this directory.
//
// Usage:
//   int main(int argc, char **argv) {
//     perf::Benchmark Bench(argc, argv, "My benchmark", "What it measures",
//                           perf::Unit::ms);
//     // ... setup ...
//     Bench.add_perf_test("my op", [&] { op(); });  // measures immediately
//     // ... verification and cleanup ...
//     return 0;
//   }
//
// Each add_perf_test() prints one line:
//   PERF_RESULT {"name":...,"variant":...,"unit":...,"value":<median>,
//                "stddev":...,"min":...,"reps":...,"lower_is_better":true,
//                "batch":...,"mad":...}

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace perf {

enum class Unit { s, ms, us, ns };

// Thrown by check().
struct CheckFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void check(bool Cond, const std::string &Msg) {
  if (!Cond)
    throw CheckFailure("correctness check failed: " + Msg);
}

// Prevents the compiler from optimizing away the computation of Value.
template <class T> inline void do_not_optimize(T &&Value) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" : : "r,m"(Value) : "memory");
#else
  static volatile char Sink;
  Sink = *reinterpret_cast<const volatile char *>(&Value);
#endif
}

// Measurement parameters of one add_perf_test(). All setters are optional.
class Opts {
public:
  // Untimed calls before sampling (the first one also calibrates the batch).
  Opts &warmup(int N) {
    Warmup = N;
    return *this;
  }
  // Minimum number of timed samples.
  Opts &reps(int N) {
    Reps = N;
    return *this;
  }
  // Keep sampling until both reps() and min_time() are reached.
  Opts &min_time(double Seconds) {
    MinTime = Seconds;
    return *this;
  }
  Opts &max_reps(int N) {
    MaxReps = N;
    return *this;
  }
  // Calibration target: each sample runs the function enough times to take
  // about this long.
  Opts &min_sample_time(double Seconds) {
    MinSampleTime = Seconds;
    return *this;
  }

  int Warmup = 1;
  int Reps = 10;
  double MinTime = 0.1;
  int MaxReps = 1000;
  double MinSampleTime = 1e-3;
};

namespace detail {

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
  if (!valid_label(S))
    throw std::invalid_argument(
        std::string(What) + " \"" + std::string(S) +
        "\" is longer than 120 characters or contains characters outside "
        "[A-Za-z0-9 _.,:;<>()=+/*&#-]");
}

inline bool valid_description(std::string_view S) {
  if (S.size() > 200)
    return false;
  for (char C : S)
    if (C < 0x20 || C > 0x7e || C == '"' || C == '\\')
      return false;
  return true;
}

inline const char *unit_name(Unit U) {
  switch (U) {
  case Unit::s:
    return "s";
  case Unit::ms:
    return "ms";
  case Unit::us:
    return "us";
  case Unit::ns:
    return "ns";
  }
  return "";
}

inline double ns_per_unit(Unit U) {
  switch (U) {
  case Unit::s:
    return 1e9;
  case Unit::ms:
    return 1e6;
  case Unit::us:
    return 1e3;
  case Unit::ns:
    return 1;
  }
  return 1;
}

inline std::string num(double V) {
  if (!std::isfinite(V))
    throw std::runtime_error("perf::Benchmark: non-finite measurement");
  char Buf[64];
  std::snprintf(Buf, sizeof(Buf), "%.9g", V);
  return Buf;
}

inline double median_sorted(const std::vector<double> &V) {
  size_t N = V.size();
  return N % 2 ? V[N / 2] : (V[N / 2 - 1] + V[N / 2]) / 2;
}

inline double seconds_since(std::chrono::steady_clock::time_point Start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - Start)
      .count();
}

} // namespace detail

class Benchmark {
public:
  Benchmark(int argc, char **argv, std::string name, std::string description,
            Unit unit)
      : Name(std::move(name)), U(unit) {
    detail::validate_label("benchmark name", Name);
    if (Name.empty())
      throw std::invalid_argument("empty benchmark name");
    if (!detail::valid_description(description))
      throw std::invalid_argument(
          "benchmark description must be printable ASCII without '\"' or "
          "'\\' and at most 200 characters");
    parse_cli(argc, argv);
    if (!List) {
      std::cout << "PERF_INFO {\"benchmark\":\"" << Name
                << "\",\"description\":\"" << description << "\",\"unit\":\""
                << detail::unit_name(U) << "\",\"variant\":\"" << Variant
                << "\"}" << std::endl;
    }
  }

  Benchmark(const Benchmark &) = delete;
  Benchmark &operator=(const Benchmark &) = delete;

  ~Benchmark() {
    // A misspelled --case would otherwise silently run nothing.
    if (List || std::uncaught_exceptions() != 0)
      return;
    for (const std::string &C : Selected)
      if (!Seen.count(C))
        fatal("no case named \"" + C + "\" (see --list)");
  }

  // Value of a --key=value command-line argument; throws if missing.
  std::string arg(const std::string &Key) const {
    auto It = Args.find(Key);
    if (It == Args.end())
      throw std::runtime_error("missing required argument --" + Key + "=...");
    return It->second;
  }

  std::string arg(const std::string &Key, const std::string &Default) const {
    auto It = Args.find(Key);
    return It == Args.end() ? Default : It->second;
  }

  // Measures Fn now and prints one PERF_RESULT line. Exceptions from Fn
  // propagate.
  template <class F>
  void add_perf_test(const std::string &name, F &&Fn, const Opts &O = Opts()) {
    detail::validate_label("test name", name);
    if (name.empty())
      throw std::logic_error("empty test name");
    if (!Seen.insert(name).second)
      throw std::logic_error("duplicate test name \"" + name + "\"");
    if (List) {
      std::cout << name << std::endl;
      return;
    }
    if (!Selected.empty() &&
        std::find(Selected.begin(), Selected.end(), name) == Selected.end())
      return;
    measure(name, Fn, O);
  }

private:
  using Clock = std::chrono::steady_clock;

  [[noreturn]] static void fatal(const std::string &Msg) {
    std::cerr << "perf_harness: error: " << Msg << std::endl;
    std::exit(2);
  }

  static std::optional<int> parse_int(const std::string &S, int Min) {
    try {
      size_t Pos = 0;
      int V = std::stoi(S, &Pos);
      if (Pos == S.size() && V >= Min)
        return V;
    } catch (...) {
    }
    return std::nullopt;
  }

  static std::optional<double> parse_double(const std::string &S) {
    try {
      size_t Pos = 0;
      double V = std::stod(S, &Pos);
      if (Pos == S.size() && std::isfinite(V) && V >= 0)
        return V;
    } catch (...) {
    }
    return std::nullopt;
  }

  void parse_cli(int argc, char **argv) {
    for (int I = 1; I < argc; ++I) {
      std::string A = argv[I];
      if (A == "--list") {
        List = true;
        continue;
      }
      size_t Eq = A.find('=');
      if (A.rfind("--", 0) != 0 || Eq == std::string::npos || Eq == 2)
        fatal("unexpected argument \"" + A +
              "\"; expected --list or --key=value");
      std::string Key = A.substr(2, Eq - 2), Value = A.substr(Eq + 1);
      if (Key == "case") {
        Selected.push_back(Value);
      } else if (Key == "reps") {
        if (!(Reps = parse_int(Value, 1)))
          fatal("--reps must be an integer >= 1");
      } else if (Key == "warmup") {
        if (!(Warmup = parse_int(Value, 0)))
          fatal("--warmup must be an integer >= 0");
      } else if (Key == "min-time") {
        if (!(MinTime = parse_double(Value)))
          fatal("--min-time must be a non-negative number of seconds");
      } else if (Key == "variant") {
        if (!detail::valid_label(Value))
          fatal("--variant is longer than 120 characters or contains "
                "characters outside [A-Za-z0-9 _.,:;<>()=+/*&#-]");
        Variant = Value;
      } else {
        Args[Key] = Value;
      }
    }
  }

  template <class F>
  void measure(const std::string &TestName, F &Fn, const Opts &O) {
    const double NsPerUnit = detail::ns_per_unit(U);
    const int Warmup = Warmup_or(O);
    const bool Exact = Reps.has_value();
    const int MinReps = Exact ? *Reps : O.Reps;
    const int MaxReps = Exact ? *Reps : std::max(O.MaxReps, O.Reps);
    const double MinTimeS = MinTime.value_or(O.MinTime);

    // Calibration (counts as the first warm-up): grow the batch size K until
    // one batch takes at least MinSampleTime. Timing a single cold call would
    // underestimate K for fast tests, because of first-call and clock costs.
    constexpr long long MaxBatch = 10000000;
    long long K = 1;
    for (;;) {
      double Elapsed = run_batch(Fn, K);
      if (Elapsed >= O.MinSampleTime || K >= MaxBatch)
        break;
      long long Next = 2 * K;
      if (Elapsed > 0)
        Next = std::max(Next, static_cast<long long>(
                                  std::ceil(K * O.MinSampleTime / Elapsed)));
      K = std::min(Next, MaxBatch);
    }

    for (int I = 1; I < Warmup; ++I)
      run_batch(Fn, K);

    std::vector<double> Samples;
    double Total = 0;
    while (static_cast<int>(Samples.size()) < MaxReps) {
      if (static_cast<int>(Samples.size()) >= MinReps &&
          (Exact || Total >= MinTimeS))
        break;
      double Elapsed = run_batch(Fn, K);
      Total += Elapsed;
      Samples.push_back(Elapsed * 1e9 / NsPerUnit / static_cast<double>(K));
    }

    std::vector<double> V = Samples;
    std::sort(V.begin(), V.end());
    const size_t N = V.size();
    const double Median = detail::median_sorted(V);
    double Mean = 0;
    for (double X : V)
      Mean += X;
    Mean /= N;
    double Var = 0;
    for (double X : V)
      Var += (X - Mean) * (X - Mean);
    const double Stddev = N > 1 ? std::sqrt(Var / (N - 1)) : 0;
    std::vector<double> Dev;
    for (double X : V)
      Dev.push_back(std::fabs(X - Median));
    std::sort(Dev.begin(), Dev.end());
    const double Mad = detail::median_sorted(Dev);

    std::string Line =
        "PERF_RESULT {\"name\":\"" + TestName + "\",\"variant\":\"" + Variant +
        "\",\"unit\":\"" + detail::unit_name(U) +
        "\",\"value\":" + detail::num(Median) +
        ",\"stddev\":" + detail::num(Stddev) +
        ",\"min\":" + detail::num(V.front()) +
        ",\"reps\":" + std::to_string(N) +
        ",\"lower_is_better\":true,\"batch\":" + std::to_string(K) +
        ",\"mad\":" + detail::num(Mad) + "}";
    std::cout << Line << std::endl;
  }

  int Warmup_or(const Opts &O) const { return Warmup.value_or(O.Warmup); }

  // Runs Fn K times and returns the elapsed seconds.
  template <class F> static double run_batch(F &Fn, long long K) {
    auto Start = Clock::now();
    for (long long I = 0; I < K; ++I)
      Fn();
    return detail::seconds_since(Start);
  }

  std::string Name;
  Unit U;
  std::string Variant;
  bool List = false;
  std::vector<std::string> Selected;
  std::optional<int> Reps, Warmup;
  std::optional<double> MinTime;
  std::map<std::string, std::string> Args;
  std::set<std::string> Seen;
};

} // namespace perf
