import ast
p = '/opt/vllm/vllm/v1/worker/gpu/model_runner.py'
src = open(p).read()

old = '''    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)'''
new = '''    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        # site-21 (experiment): no-op the consume — the ring's appends double-
        # counted tokens on non-last ranks (context duplication -> garbage
        # text) regardless of kernel implementation. Testing whether non-last
        # ranks need the sync at all in this configuration.
        return
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)'''
n = src.count(old)
assert n == 1, f'anchor count {n}'
src = src.replace(old, new)
ast.parse(src)
open(p, 'w').write(src)
print('site-21 applied, syntax OK')
