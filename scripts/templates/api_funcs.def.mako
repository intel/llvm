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
%for obj in th.get_loader_functions(specs, meta, n, tags):
%if n + "Loader" in obj:
_UR_API(${obj})
%endif
%endfor
