#!/usr/bin/env python3
"""site-26: PP draft-token table sync (the v37 fork-design fix).

propose() runs only on the LAST PP rank, but every rank's prepare_inputs
gather (combine_sampled_and_draft_tokens) reads its own req_states.draft_tokens.
Non-last ranks were feeding never-written zero rows at draft positions ->
garbage verification on every spec step (v37's 'every accepted draft is garbage').

Fix: third broadcast on the existing site-17/18 ring. Last rank sends the FULL
[max_num_reqs, num_spec] table right after propose; non-last ranks receive it
in-place at their existing ring receive(). Full-table semantics match the
pp_size-step decode cadence (next_decode_eligible_step): rows of requests
absent from this step keep their latest proposals on every rank.
Both sides gated on the SAME predicate (num_speculative_steps > 0 and pp>1)
-- unconditional symmetry, the site-18 lesson.
"""
import ast, sys

PP = "/opt/vllm/vllm/v1/worker/gpu/pp_utils.py"
MR = "/opt/vllm/vllm/v1/worker/gpu/model_runner.py"

def patch(path, edits, must_apply_all=True):
    src = open(path).read()
    for i, (old, new) in enumerate(edits):
        if new in src and old not in src:
            print(f"[{path}] edit {i}: already applied")
            continue
        n = src.count(old)
        assert n == 1, f"[{path}] edit {i}: anchor count {n} (need 1)"
        src = src.replace(old, new)
        print(f"[{path}] edit {i}: applied")
    ast.parse(src)
    open(path, "w").write(src)
    ast.parse(open(path).read())
    print(f"[{path}] ast OK")

# ---------- pp_utils.py ----------
pp_edits = [
    # A1: trace counter init
    (
        "    def on_req_idx_freed(self, req_idx: int) -> None:",
        "        self._draft_trace_count = 0\n\n"
        "    def on_req_idx_freed(self, req_idx: int) -> None:",
    ),
    # A2: receive signature
    (
        "    def receive(self, input_batch: InputBatch) -> bool:",
        "    def receive(\n"
        "        self, input_batch: InputBatch, draft_recv_buf: torch.Tensor | None = None\n"
        "    ) -> bool:",
    ),
    # A3: third broadcast inside receive (anchored unique via the Event that follows)
    (
        "        torch.distributed.broadcast(\n"
        "            combined, src=self.last_rank, group=self.broadcast_group\n"
        "        )\n"
        "        event = torch.cuda.Event()",
        "        torch.distributed.broadcast(\n"
        "            combined, src=self.last_rank, group=self.broadcast_group\n"
        "        )\n"
        "        # site-26: receive the last rank's draft-token table (full buffer,\n"
        "        # in-place). Serialized after the sampled/combined broadcasts on\n"
        "        # the same group, so this recv lands only after the last rank's\n"
        "        # post-sample propose() has enqueued its send.\n"
        "        if draft_recv_buf is not None:\n"
        "            torch.distributed.broadcast(\n"
        "                draft_recv_buf, src=self.last_rank, group=self.broadcast_group\n"
        "            )\n"
        "            if self._draft_trace_count < 5:\n"
        "                self._draft_trace_count += 1\n"
        "                row = int(input_batch.idx_mapping_np[0])\n"
        "                print(\n"
        "                    f\"TRACE DRAFTSYNC pid={os.getpid()} \"\n"
        "                    f\"n={self._draft_trace_count} row{row}=\"\n"
        "                    f\"{draft_recv_buf[row][:3].tolist()}\",\n"
        "                    flush=True,\n"
        "                )\n"
        "        event = torch.cuda.Event()",
    ),
    # A4: broadcast_drafts method
    (
        "    def broadcast(\n"
        "        self,\n"
        "        sampled_token_ids: torch.Tensor,",
        "    def broadcast_drafts(self, draft_tokens: torch.Tensor) -> None:\n"
        "        \"\"\"site-26: last-rank send of the draft-token table.\n"
        "        Called post-propose; recv side is receive(draft_recv_buf=...).\n"
        "        \"\"\"\n"
        "        assert self.is_last_rank\n"
        "        torch.distributed.broadcast(\n"
        "            draft_tokens.contiguous(),\n"
        "            src=self.last_rank,\n"
        "            group=self.broadcast_group,\n"
        "        )\n\n"
        "    def broadcast(\n"
        "        self,\n"
        "        sampled_token_ids: torch.Tensor,",
    ),
]

# ---------- model_runner.py ----------
mr_edits = [
    # B1: non-last rank passes its draft table as the recv buffer
    (
        "            all_decode_next = self.pp_handler.receive(input_batch)",
        "            all_decode_next = self.pp_handler.receive(\n"
        "                input_batch,\n"
        "                draft_recv_buf=(\n"
        "                    self.req_states.draft_tokens\n"
        "                    if self.num_speculative_steps > 0\n"
        "                    else None\n"
        "                ),\n"
        "            )",
    ),
    # B2: last rank broadcasts the table after the drafter wrote it
    (
        "        if self.num_speculative_steps > 0:\n"
        "            # Spec-decode and diffusion LLMs both use draft tokens but the latter does\n"
        "            # not have a speculator (i.e. self.speculator is None)\n"
        "            self.draft_tokens_handler.set_draft_tokens(\n"
        "                input_batch,\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
        "            )",
        "        if self.num_speculative_steps > 0:\n"
        "            # Spec-decode and diffusion LLMs both use draft tokens but the latter does\n"
        "            # not have a speculator (i.e. self.speculator is None)\n"
        "            self.draft_tokens_handler.set_draft_tokens(\n"
        "                input_batch,\n"
        "                self.req_states.draft_tokens[input_batch.idx_mapping],\n"
        "            )\n"
        "            if self.pp_handler is not None:\n"
        "                # site-26: sync the speculator's draft table to all PP ranks.\n"
        "                # propose() runs only here (last rank), but every rank's\n"
        "                # combine_sampled_and_draft_tokens gather reads its own\n"
        "                # req_states.draft_tokens -- non-last ranks were feeding\n"
        "                # zero rows at draft positions (v37 garbage verification).\n"
        "                # Same predicate as the receive side; symmetric ring.\n"
        "                self.pp_handler.broadcast_drafts(self.req_states.draft_tokens)",
    ),
]

patch(PP, pp_edits)
patch(MR, mr_edits)

# grep-verify markers
for path, markers in [
    (PP, ["broadcast_drafts", "draft_recv_buf", "TRACE DRAFTSYNC"]),
    (MR, ["broadcast_drafts(self.req_states.draft_tokens)", "draft_recv_buf="]),
]:
    s = open(path).read()
    for m in markers:
        assert m in s, f"VERIFY FAIL: {m} not in {path}"
    print(f"[{path}] markers OK")
print("site-26 PATCH COMPLETE")
