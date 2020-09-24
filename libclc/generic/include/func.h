#ifndef CLC_FUNC
#define CLC_FUNC

#define _CLC_OVERLOAD __attribute__((overloadable))
#define _CLC_DECL
#define _CLC_DEF __attribute__((always_inline))
#define _CLC_INLINE __attribute__((always_inline)) inline
#define _CLC_CONVERGENT __attribute__((convergent))
#define _CLC_PURE __attribute__((pure))
#define _CLC_CONSTFN __attribute__((const))

#endif // CLC_FUNC
