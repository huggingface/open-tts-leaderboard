import json
import time
import torch


class CompileBackend:
    def __init__(self):
        from torch._dynamo.backends.registry import lookup_backend
        self.backend = lookup_backend('inductor')
        self.events = []

    def __call__(self, gm, inputs):
        begin = time.perf_counter()
        event = dict(index=len(self.events), graph_nodes=len(list(gm.graph.nodes)),
                     input_shapes=[list(map(str, x.shape)) for x in inputs if isinstance(x, torch.Tensor)])
        self.events.append(event)
        print('COMPILE_GRAPH_START', json.dumps({k: v for k, v in event.items() if k != 'input_shapes'}), flush=True)
        try:
            result = self.backend(gm, inputs)
            event['status'] = 'compiled'
            return result
        except Exception as error:
            event['status'] = 'error'
            event['error_type'] = type(error).__name__
            event['error'] = str(error)[:1500]
            raise
        finally:
            event['backend_seconds'] = time.perf_counter() - begin
            print('COMPILE_GRAPH_END', json.dumps({k: v for k, v in event.items() if k != 'input_shapes'}), flush=True)


def compiler_counters():
    from torch._dynamo.utils import counters
    return {name: {str(k): int(v) for k, v in counters[name].items()}
            for name in ('stats', 'frames', 'graph_break', 'inductor')}
