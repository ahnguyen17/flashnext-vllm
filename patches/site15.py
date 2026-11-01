import ast
p = '/opt/vllm/vllm/v1/worker/gpu/model_runner.py'
src = open(p).read()

# --- 1) insert module-level pure-torch helper before first top-level class/def ---
helper = '''
def _post_update_pure_torch(
    idx_mapping,
    num_computed_tokens,
    last_sampled_tokens,
    sampled_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    all_token_ids,
    total_len,
) -> None:
    """site-15: non-last-PP-rank bookkeeping without triton.

    The None-specialization of _post_update_kernel deadlocks in
    cuModuleLoadData on sm_86 (patched driver) inside the multi-rank
    engine; semantics here mirror the triton kernel for the
    output_bin_counts=None path.
    """
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return
    valid = idx_mapping >= 0
    vi = idx_mapping[valid].long()
    if vi.numel() == 0:
        return
    ns = num_sampled[valid].long()
    sv = sampled_tokens[valid]
    old_tot = total_len[vi].long()

    m = ns > 0
    rows = vi[m]
    if rows.numel():
        last_sampled_tokens[rows] = sv[m, ns[m] - 1].to(last_sampled_tokens.dtype)
        total_len[rows] = (old_tot[m] + ns[m]).to(total_len.dtype)

    total_new = int(ns.sum().item())
    if total_new > 0:
        vr = torch.repeat_interleave(
            torch.arange(vi.shape[0], device=idx_mapping.device), ns
        )
        within = torch.arange(total_new, device=idx_mapping.device) - torch.repeat_interleave(
            torch.cumsum(ns, 0) - ns, ns
        )
        dest = old_tot[vr] + within
        all_token_ids[vi[vr], dest] = sv[vr, within].to(all_token_ids.dtype)

    if query_start_loc is not None:
        qlen = (query_start_loc[1:] - query_start_loc[:-1]).long()
        delta = qlen[valid] - num_rejected[valid].long()
        nz = delta != 0
        if vi[nz].numel():
            num_computed_tokens[vi[nz]] += delta[nz].to(num_computed_tokens.dtype)


'''
lines = src.split('\n')
insert_at = next(i for i, l in enumerate(lines) if l.startswith('class ') or (l.startswith('def ') and not l.startswith('def _')))
lines.insert(insert_at, helper.rstrip('\n'))
src = '\n'.join(lines)

# --- 2) branch the call: last rank keeps triton, others take pure-torch ---
old_call = '''        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None
        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
        )'''
new_call = '''        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
            post_update(
                idx_mapping,
                self.req_states.num_computed_tokens.gpu,
                self.req_states.last_sampled_tokens,
                output_bin_counts,
                sampled_tokens,
                num_sampled,
                num_rejected,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.total_len.gpu,
            )
        else:
            # site-15: pure-torch path on non-last PP ranks — see
            # _post_update_pure_torch docstring.
            _post_update_pure_torch(
                idx_mapping,
                self.req_states.num_computed_tokens.gpu,
                self.req_states.last_sampled_tokens,
                sampled_tokens,
                num_sampled,
                num_rejected,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.total_len.gpu,
            )'''
n = src.count(old_call)
assert n == 1, f'call anchor count {n}'
src = src.replace(old_call, new_call)

ast.parse(src)
open(p, 'w').write(src)
print('site-15 applied, syntax OK')
