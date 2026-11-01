import ast
p = '/opt/vllm/vllm/v1/worker/gpu/model_runner.py'
src = open(p).read()

old = '''    if query_start_loc is not None:
        qlen = (query_start_loc[1:] - query_start_loc[:-1]).long()
        delta = qlen[valid] - num_rejected[valid].long()
        nz = delta != 0
        if vi[nz].numel():
            num_computed_tokens[vi[nz]] += delta[nz].to(num_computed_tokens.dtype)'''
new = '''    # site-19: match the triton kernel exactly — when query_start_loc is None
    # (the PP ring path), the kernel still applies delta = 0 - num_rejected.
    # The old port skipped the adjustment, inflating num_computed on non-last
    # ranks by the rejection count each step -> context desync -> garbage text.
    if query_start_loc is not None:
        qlen = (query_start_loc[1:] - query_start_loc[:-1]).long()
    else:
        qlen = torch.zeros(num_reqs, dtype=torch.long, device=idx_mapping.device)
    delta = qlen[valid] - num_rejected[valid].long()
    nz = delta != 0
    if vi[nz].numel():
        num_computed_tokens[vi[nz]] += delta[nz].to(num_computed_tokens.dtype)'''
n = src.count(old)
assert n == 1, f'anchor count {n}'
src = src.replace(old, new)
ast.parse(src)
open(p, 'w').write(src)
print('site-19 applied, syntax OK')
