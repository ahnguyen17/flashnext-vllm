import ast
p = '/opt/vllm/vllm/v1/worker/gpu/model_runner.py'
src = open(p).read()

old = '''        else:
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
new = '''        else:
            # site-20: non-last PP ranks use the reference triton kernel with a
            # dummy histogram tensor (the None-variant's module load is what
            # deadlocked; the tensor-variant loads fine). Histogram updates
            # land in the throwaway buffer — non-last ranks don't sample.
            if self._pp_dummy_bin_counts is None:
                vocab = self.model_config.get_vocab_size()
                self._pp_dummy_bin_counts = torch.zeros(
                    idx_mapping.shape[0], vocab,
                    dtype=torch.int32, device=idx_mapping.device,
                ) if idx_mapping.shape[0] > 0 else torch.zeros(
                    1, vocab, dtype=torch.int32, device=idx_mapping.device,
                )
            post_update(
                idx_mapping,
                self.req_states.num_computed_tokens.gpu,
                self.req_states.last_sampled_tokens,
                self._pp_dummy_bin_counts[: idx_mapping.shape[0]],
                sampled_tokens,
                num_sampled,
                num_rejected,
                query_start_loc,
                self.req_states.all_token_ids.gpu,
                self.req_states.total_len.gpu,
            )'''
n = src.count(old)
assert n == 1, f'anchor count {n}'
src = src.replace(old, new)

# init the dummy buffer attribute in __init__ (anchor: pp_handler creation)
anchor = '        self.pp_handler: PPHandler | None = None'
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + '\n        self._pp_dummy_bin_counts = None  # site-20')

ast.parse(src)
open(p, 'w').write(src)
print('site-20 applied, syntax OK')
