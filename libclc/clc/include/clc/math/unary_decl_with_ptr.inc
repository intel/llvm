_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                     global __CLC_GENTYPE *ptr);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_FUNCTION(__CLC_GENTYPE x,
                                                     local __CLC_GENTYPE *ptr);
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE
__CLC_FUNCTION(__CLC_GENTYPE x, private __CLC_GENTYPE *ptr);

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
_CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE
__CLC_FUNCTION(__CLC_GENTYPE x, generic __CLC_GENTYPE *ptr);
#endif
