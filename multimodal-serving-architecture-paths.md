# Multimodal Serving on llm-d: Challenges and Paths Forward

## Why This Matters

Any-to-any multimodal models run as multi-stage pipelines. [Qwen3-Omni](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/pipeline.py) chains a thinker LLM (stage 0, autoregressive), a talker codec generator (stage 1, autoregressive), and an audio decoder (stage 2, single forward pass). vllm-omni supports [31 such pipelines](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/config/pipeline_registry.py) across TTS, omni-modal, and image generation.

These stages have different compute profiles. The thinker is memory-bandwidth bound (AR token generation); the decoder is compute-bound (single-pass waveform synthesis). Scaling the thinker to handle load also scales the decoder, wasting GPUs on a stage that isn't the bottleneck. Per-stage scaling, failure isolation, and intelligent routing require llm-d to manage stages independently.

Today vllm-omni's [orchestrator](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py) owns stage sequencing, inter-stage transforms, [async-chunk streaming](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_chunk.md), and replica management. llm-d handles none of it.

## Why Stages Cannot Be Isolated Today

**The orchestrator owns the data contract between stages.** The orchestrator calls [`process_engine_inputs()`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py#L1284) to transform one stage's output into the next stage's input using [per-model transform functions](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/model_executor/stage_input_processors). These operate directly on engine-internal types:

```python
# stage_input_processors/qwen3_tts.py — talker2code2wav
output = talker_output.outputs[0]                          # EngineCoreOutput
audio_codes = output.multimodal_output["codes"]["audio"].to(torch.long)  # raw tensor
```

Code2wav never sees this. It receives pre-transformed [`OmniTokensPrompt`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/inputs/data.py) objects built by the orchestrator. No serialization contract exists between stages.

**Transforms run outside the consuming stage.** The non-async path runs in the [orchestrator thread](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py#L1284). The async-chunk and full-payload paths run in the [upstream worker's connector mixin](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/worker/omni_connector_model_runner_mixin.py). Neither the producing stage nor the consuming stage owns the transform.

**Async-chunk streaming spans both stages.** The orchestrator [pre-warms downstream stages](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py#L1329) with placeholder tokens before upstream finishes. The worker mixin manages frame accumulation and dynamic chunk sizing based on the downstream stage's `scheduler_max_num_seqs`. This cross-stage state reduces TTFP by [~92% on Qwen3-Omni](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_chunk.md) (6459ms to 523ms at concurrency 1).

**Stages cannot boot independently.** Only the thinker (stage 0) owns the tokenizer ([`owns_tokenizer=True`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/model_executor/models/qwen3_omni/pipeline.py#L29)). Talker and code2wav depend on the orchestrator for tokenized input, weight selection, and engine initialization.

## Paths Forward

### Stage-Level Disaggregation

Each stage runs as an independent server. Large inter-stage data moves peer-to-peer via connectors (Mooncake/NIXL). The coordinator forwards transfer handles between stages, the same pattern as KV cache transfer in P/D disaggregation.

Requires changes in vllm-omni:
- **Serialization boundary.** Each stage defines its own input/output contract. The consuming stage accepts serialized upstream output and applies its own transform on ingestion, rather than the orchestrator doing it.
- **Standalone boot.** Stages that lack tokenizers (talker, code2wav) need an initialization path that handles weight loading and engine setup without the orchestrator.
- **Stage-level streaming.** The talker streams frames directly to code2wav (stage-to-stage, coordinator provides the address but is not in the data path). Dynamic chunk sizing moves into the talker as a per-request parameter.

The coordinator stays stateless and model-unaware. Its config is just a step list:

```yaml
pipeline:
  steps: [thinker, talker, code2wav]
```

Unlike P/D disaggregation where the coordinator constructs [protocol-specific flags](https://github.com/llm-d/coordinator/blob/main/pkg/connectors/kv/nixlv2.go) (`do_remote_prefill`, `do_remote_decode`) because the same pod can operate in different modes per-request, standalone omni stages always know their role. The coordinator just forwards the transfer handle returned by each stage.

For Qwen3-Omni the request flow would look like:

```
Client POST /v1/audio/speech
  → Coordinator sets EPP-Phase: thinker, EPP picks thinker pod
  → Thinker generates, writes latent to connector, returns handle
  → Coordinator sets EPP-Phase: talker, EPP picks talker pod
  → Forwards handle. Talker pulls latent peer-to-peer, generates codec, returns handle
  → Coordinator sets EPP-Phase: code2wav, EPP picks decoder pod
  → Forwards handle. Decoder pulls codec peer-to-peer, returns audio
  → Coordinator proxies audio to client
```

The coordinator never parses stage output. It forwards handles the same way it forwards [KV transfer params in P/D disaggregation today](https://github.com/llm-d/coordinator/blob/main/pkg/steps/prefill.go). EPP role filters per stage type follow the existing [prefill/decode filter pattern](https://github.com/llm-d/llm-d-router/blob/main/pkg/epp/framework/plugins/scheduling/filter/bylabel/roles.go). Each stage gets its own scheduling profile with stage-appropriate scorers:

```yaml
schedulingProfiles:
- name: thinker   # prefix-cache-scorer, kv-cache-utilization-scorer
- name: talker    # queue-scorer
- name: code2wav  # queue-scorer, running-requests-size-scorer
```

### Opaque Group Routing

vllm-omni groups are opaque to llm-d. The orchestrator handles stage sequencing and per-stage [`num_replicas`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/config/stage_config.py#L440) internally. llm-d picks the best group for each request, autoscales the number of groups, and handles group failures. No coordinator involvement, no pipeline awareness in llm-d.

Requires vllm-omni to expose pipeline-level metrics (queue depth, per-stage utilization, pipeline latency) for EPP to consume. Requires new EPP plugins for omni-modal routing (speaker affinity, passthrough parser for audio responses).

All 31 topologies work without changes to the orchestrator or coordinator. Trades per-stage control for a clean boundary and minimal investment.

## Comparison

|  | Stage-Level Disaggregation | Opaque Group Routing |
|--|--|--|
| Per-stage scaling | Yes | Per-group |
| Async-chunk streaming | Reimplemented as stage-to-stage | Kept |
| vllm-omni changes | Significant (serialization contracts, standalone boot, stage streaming) | Minimal (metrics API) |
| llm-d changes | Minimal (handle forwarding, role filters) | Minimal (EPP plugins) |
| GPU efficiency | Best | Middle |
