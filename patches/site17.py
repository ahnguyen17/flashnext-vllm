import ast
p = '/opt/vllm/vllm/v1/worker/gpu/pp_utils.py'
src = open(p).read()

# --- receive(): main-stream broadcasts (drop side-stream dance) ---
old_recv = '''        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)'''
new_recv = '''        num_reqs = input_batch.num_reqs
        # site-17: run the ring on the MAIN stream. The side-stream +
        # sibling-communicator interleaving deadlocks against the inter-stage
        # p2p sends under MTP spec scheduling (NCCL enqueued-never-started on
        # ranks 0/1 while rank 2's part waits for peers).
        sampled_tokens = torch.empty(
            num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
        )
        combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
        torch.distributed.broadcast(
            sampled_tokens, src=self.last_rank, group=self.broadcast_group
        )
        torch.distributed.broadcast(
            combined, src=self.last_rank, group=self.broadcast_group
        )
        event = torch.cuda.Event()
        event.record(self.main_stream)
        num_sampled, num_rejected = combined.unbind(dim=0)'''
n = src.count(old_recv)
assert n == 1, f'recv anchor count {n}'
src = src.replace(old_recv, new_recv)

# --- broadcast(): main-stream ---
old_bc = '''        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)'''
new_bc = '''        # site-17: main-stream ring (see receive() note).
        torch.distributed.broadcast(
            sampled_token_ids.contiguous(),
            src=self.last_rank,
            group=self.broadcast_group,
        )
        combined = torch.stack((num_sampled, num_rejected), dim=0)
        torch.distributed.broadcast(
            combined, src=self.last_rank, group=self.broadcast_group
        )'''
n = src.count(old_bc)
assert n == 1, f'broadcast anchor count {n}'
src = src.replace(old_bc, new_bc)

ast.parse(src)
open(p, 'w').write(src)
print('site-17 applied, syntax OK')
