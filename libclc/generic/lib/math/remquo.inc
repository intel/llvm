_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE remquo(__CLC_GENTYPE x, __CLC_GENTYPE y, __CLC_ADDRESS_SPACE __CLC_INTN *q) {
  private __CLC_INTN private_q;
  __CLC_GENTYPE ret = __clc_remquo(x, y, &private_q);
  *q = private_q;
  return ret;
}
