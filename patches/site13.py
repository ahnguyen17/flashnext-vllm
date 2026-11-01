import ast
p='/opt/vllm/vllm/v1/executor/multiproc_executor.py'
src=open(p).read()
anchor='            rank = kwargs.get("rank", 0)'
n=src.count(anchor)
assert n==1, f'anchor count {n}'
patch=anchor+'''
            # site-13: per-rank TRITON_CACHE_DIR. On mixed-arch rigs (sm_86 +
            # sm_80) a shared cache lets one rank load another rank's cubin,
            # and cuModuleLoadData spins forever on the wrong-arch binary.
            try:
                _tc = os.environ.get("TRITON_CACHE_DIR", "/root/.cache/vllm/triton")
                os.environ["TRITON_CACHE_DIR"] = f"{_tc}-rank{rank}"
            except Exception:
                pass'''
open(p,'w').write(src.replace(anchor,patch))
ast.parse(open(p).read())
print('site-13 applied, syntax OK')
