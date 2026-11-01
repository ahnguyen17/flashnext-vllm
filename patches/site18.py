import ast
p = '/opt/vllm/vllm/v1/worker/gpu/pp_utils.py'
src = open(p).read()

# --- receive(): always participate; mask becomes a data filter only ---
old_r = '''    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False
'''
new_r = '''    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        # site-18: the ring must be UNCONDITIONALLY symmetric on all ranks.
        # Rank-local mask divergence (pipeline state lag) made the last rank
        # skip broadcasts that non-last ranks were blocked in -> NCCL
        # enqueued-never-started deadlock on the first spec step. The mask is
        # now only a per-request data filter at consume time.
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            need_sampled_mask = np.zeros(input_batch.num_reqs, dtype=bool)
'''
n = src.count(old_r)
assert n == 1, f'receive anchor {n}'
src = src.replace(old_r, new_r)

# --- broadcast(): always participate; pad send shape to max_sample_len ---
old_b = '''    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
'''
new_b = '''    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        # site-18: always join the ring (see receive() note). Pad the token
        # tensor to max_sample_len so numel matches the receivers' buffer;
        # padded -1 entries are skipped by the consumers' filtering.
        assert sampled_token_ids.dtype == torch.int64
        if sampled_token_ids.shape[-1] < self.max_sample_len:
            pad = torch.full(
                (sampled_token_ids.shape[0],
                 self.max_sample_len - sampled_token_ids.shape[-1]),
                -1, dtype=sampled_token_ids.dtype, device=sampled_token_ids.device,
            )
            sampled_token_ids = torch.cat([sampled_token_ids, pad], dim=-1)
        elif sampled_token_ids.shape[-1] > self.max_sample_len:
            sampled_token_ids = sampled_token_ids[:, : self.max_sample_len]
'''
n = src.count(old_b)
assert n == 1, f'broadcast anchor {n}'
src = src.replace(old_b, new_b)

if 'import numpy as np' not in src:
    assert 'import numpy as np' in src or 'from numpy' in src, 'numpy import missing!'

ast.parse(src)
open(p, 'w').write(src)
print('site-18 applied, syntax OK')
