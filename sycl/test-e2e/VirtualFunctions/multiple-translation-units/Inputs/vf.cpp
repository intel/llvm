#include "declarations.hpp"

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
void BaseIncrement::increment(int *Data) { *Data += 1 + Mod; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
void IncrementBy2::increment(int *Data) { *Data += 2 + Mod; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
void IncrementBy4::increment(int *Data) { *Data += 4 + Mod; }

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
void IncrementBy8::increment(int *Data) { *Data += 8 + Mod; }
