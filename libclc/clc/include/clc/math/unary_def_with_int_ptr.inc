#include <clc/utils.h>

#ifndef __CLC_FUNCTION
#define __CLC_FUNCTION(x) __CLC_CONCAT(__clc_, x)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              private __CLC_INTN *iptr) {
  return __CLC_FUNCTION(FUNCTION)(x, iptr);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              global __CLC_INTN *iptr) {
  return __CLC_FUNCTION(FUNCTION)(x, iptr);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              local __CLC_INTN *iptr) {
  return __CLC_FUNCTION(FUNCTION)(x, iptr);
}

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              generic __CLC_INTN *iptr) {
  return __CLC_FUNCTION(FUNCTION)(x, iptr);
}
#endif
