#ifndef __CLC_BODY_U
#define __CLC_BODY_U __CLC_BODY
#endif
#ifndef __CLC_BODY_S
#define __CLC_BODY_S __CLC_BODY
#endif

#define __CLC_GENTYPE int
#include __CLC_BODY_S
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int2
#include __CLC_BODY_S
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int3
#include __CLC_BODY_S
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int4
#include __CLC_BODY_S
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int8
#include __CLC_BODY_S
#undef __CLC_GENTYPE

#define __CLC_GENTYPE int16
#include __CLC_BODY_S
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint
#include __CLC_BODY_U
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint2
#include __CLC_BODY_U
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint3
#include __CLC_BODY_U
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint4
#include __CLC_BODY_U
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint8
#include __CLC_BODY_U
#undef __CLC_GENTYPE

#define __CLC_GENTYPE uint16
#include __CLC_BODY_U
#undef __CLC_GENTYPE

#undef __CLC_BODY_U
#undef __CLC_BODY_S
