#include <utils.h>

#ifndef __CLC_SW_FUNC
#define __CLC_SW_FUNC __CLC_XCONCAT(__clc_, __CLC_FUNC)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_FUNC(__CLC_GENTYPE x) {
  return __CLC_SW_FUNC(x);
}
