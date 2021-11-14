#pragma once

#include <random>

inline static std::string getRandomString() {
  std::random_device Dev;
  std::mt19937 Range(Dev());
  std::uniform_int_distribution<std::mt19937::result_type> Dist(1, 255);

  size_t Size = Dist(Range);
  std::string Result = "";
  Result.resize(Size);

  for (char &C : Result) {
    C = static_cast<char>(Dist(Range));
  }

  return Result;
}
