<%!
import re
from templates import helper as th
%><%
    n=namespace
    N=n.upper()

    x=tags['$x']
    X=x.upper()
%>
%for tbl in th.get_pfntables(specs, meta, n, tags):
%for obj in tbl['functions']:
_UR_API(${th.make_func_name(n, tags, obj)})
%endfor
%endfor
