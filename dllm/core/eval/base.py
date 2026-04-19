"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

import accelerate
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

import dllm
from dllm.core.samplers import BaseSampler, BaseSamplerConfig
from dllm.utils.configs import ModelArguments


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: device and batch_size."""

    pretrained: str = ""
    device: str = "cuda"
    batch_size: int = 1
    save_generation_records_path: str = ""
    save_generation_traces: bool = False
    generation_trace_max_steps: int = 64
    save_sampler_diagnostics: bool = False

    def get_model_config(self, pretrained: str):
        """Optional: return custom model config for loading. Default None (use checkpoint config)."""
        return None


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood (and optionally loglikelihood_rolling);
    generate_until is implemented here and uses sampler + sampler_config.
    """

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        """Build a dataclass *config_cls* by copying fields from *source*, with *kwargs* overrides."""
        init = {}
        for f in dataclasses.fields(config_cls):
            if f.name in kwargs:
                init[f.name] = kwargs[f.name]
            elif hasattr(source, f.name):
                init[f.name] = getattr(source, f.name)
        return config_cls(**init)

    def __init__(
        self,
        eval_config: BaseEvalConfig | None = None,
        model_args: ModelArguments | None = None,
        sampler_config: BaseSamplerConfig | None = None,
        sampler_cls: type[BaseSampler] | None = None,
        **kwargs,
    ):
        super().__init__()
        eval_config = eval_config or BaseEvalConfig()
        # Ensure model path is in kwargs and we have a safe default for ModelArguments(__post_init__).
        model_args = model_args or ModelArguments(
            model_name_or_path=kwargs.get("pretrained")
        )
        device = kwargs.get("device", eval_config.device)

        # ── Distributed ──────────────────────────────────────────
        accelerator = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = dllm.utils.get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = dllm.utils.get_tokenizer(self.model_args)
        if sampler_config is not None:
            self.sampler_config = self._build_config(
                type(sampler_config), sampler_config, kwargs
            )
        if sampler_cls is not None:
            self.sampler = sampler_cls(model=self.model, tokenizer=self.tokenizer)

        # ── Device placement ─────────────────────────────────────
        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))
        self.save_generation_records_path = str(
            kwargs.get(
                "save_generation_records_path",
                eval_config.save_generation_records_path,
            )
            or ""
        )
        self.save_generation_traces = bool(
            kwargs.get("save_generation_traces", eval_config.save_generation_traces)
        )
        self.generation_trace_max_steps = int(
            kwargs.get(
                "generation_trace_max_steps", eval_config.generation_trace_max_steps
            )
        )
        self.save_sampler_diagnostics = bool(
            kwargs.get(
                "save_sampler_diagnostics", eval_config.save_sampler_diagnostics
            )
        )
        self._generation_record_path: Path | None = None
        if self.rank == 0 and self.save_generation_records_path:
            self._generation_record_path = Path(self.save_generation_records_path)
            self._generation_record_path.parent.mkdir(parents=True, exist_ok=True)
            self._generation_record_path.write_text("", encoding="utf-8")

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format chat history for input to the LM."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @staticmethod
    def _trim_by_stop(answer: str, stop_sequences: list[str] | tuple[str, ...]) -> str:
        for stop_seq in stop_sequences:
            if stop_seq and stop_seq in answer:
                answer = answer.split(stop_seq)[0]
        return answer

    @staticmethod
    def _extract_question_text(context: str) -> str:
        patterns = [r"\nQ:\s*", r"\nQuestion:\s*", r"\n问题：", r"\n问题:"]
        last_start = -1
        for pattern in patterns:
            match = list(re.finditer(pattern, context, flags=re.IGNORECASE))
            if match:
                last_start = max(last_start, match[-1].start())
        if last_start >= 0:
            snippet = context[last_start:].strip()
            answer_markers = ["\nA:", "\nAnswer:", "\n答案：", "\n答案:"]
            cut = len(snippet)
            for marker in answer_markers:
                pos = snippet.find(marker)
                if pos != -1:
                    cut = min(cut, pos)
            return snippet[:cut].strip()
        return context.strip()

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""

        patterns = [
            r"####\s*(.+)",
            r"\\boxed\{([^{}]+)\}",
            r"(?:final answer|answer is)\s*[:：]?\s*(.+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, stripped, flags=re.IGNORECASE)
            if matches:
                return matches[-1].strip()

        non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if non_empty_lines:
            return non_empty_lines[-1]
        return stripped

    @classmethod
    def _compress_trace(
        cls,
        trace_entries: list[dict[str, str | int]],
        max_steps: int,
    ) -> list[dict[str, str | int]]:
        if max_steps <= 0 or len(trace_entries) <= max_steps:
            return trace_entries

        keep = {0, len(trace_entries) - 1}
        if max_steps > 2:
            stride = (len(trace_entries) - 1) / float(max_steps - 1)
            for i in range(1, max_steps - 1):
                keep.add(round(i * stride))
        return [trace_entries[i] for i in sorted(keep)]

    def _build_generation_traces(
        self,
        *,
        histories: list[torch.Tensor] | None,
        prompt_token_lists: list[list[int]],
        stop_sequences: list[str] | tuple[str, ...],
    ) -> list[list[dict[str, str | int]]]:
        if not histories:
            return [[] for _ in prompt_token_lists]

        per_sample: list[list[dict[str, str | int]]] = [
            [] for _ in range(len(prompt_token_lists))
        ]
        last_texts = [None for _ in range(len(prompt_token_lists))]

        for step_idx, history in enumerate(histories):
            decoded = dllm.utils.sample_trim(
                self.tokenizer,
                history.tolist(),
                prompt_token_lists,
            )
            for sample_idx, text in enumerate(decoded):
                text = self._trim_by_stop(text, stop_sequences)
                if text != last_texts[sample_idx]:
                    per_sample[sample_idx].append({"step": step_idx, "text": text})
                    last_texts[sample_idx] = text

        return [
            self._compress_trace(trace, self.generation_trace_max_steps)
            for trace in per_sample
        ]

    def _append_generation_records(
        self,
        *,
        batch_start: int,
        contexts: tuple[str, ...],
        answers: list[str],
        gen_kwargs_list: tuple[dict, ...],
        traces: list[list[dict[str, str | int]]] | None,
        sampler_diagnostics: dict | None,
    ) -> None:
        if self.rank != 0 or self._generation_record_path is None:
            return

        with self._generation_record_path.open("a", encoding="utf-8") as writer:
            for offset, (context, answer, gen_kwargs) in enumerate(
                zip(contexts, answers, gen_kwargs_list)
            ):
                record = {
                    "index": batch_start + offset,
                    "question": self._extract_question_text(context),
                    "prompt": context,
                    "response": answer,
                    "predicted_final_answer": self._extract_final_answer(answer),
                    "stop_sequences": list(gen_kwargs.get("until", [])),
                }
                if traces is not None:
                    record["generation_trace"] = traces[offset]
                if self.save_sampler_diagnostics and sampler_diagnostics is not None:
                    if (
                        isinstance(sampler_diagnostics, dict)
                        and "per_sample" in sampler_diagnostics
                        and offset < len(sampler_diagnostics["per_sample"])
                    ):
                        record["sampler_diagnostics"] = sampler_diagnostics["per_sample"][
                            offset
                        ]
                    else:
                        record["sampler_diagnostics"] = sampler_diagnostics
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Unified generate_until scaffolding ────────────────────────────

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        out: list[str] = []

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size), desc="Generating..."
        ):
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

            prompts = [
                torch.tensor(
                    self.tokenizer(ctx)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                for ctx in contexts
            ]
            prompt_token_lists = [p.tolist() for p in prompts]

            if self.save_generation_traces:
                sampler_output = self.sampler.sample(
                    inputs=prompts,
                    config=self.sampler_config,
                    return_dict=True,
                )
                generated_ids = sampler_output.sequences
                traces = self._build_generation_traces(
                    histories=sampler_output.histories,
                    prompt_token_lists=prompt_token_lists,
                    stop_sequences=gen_kwargs_list[0]["until"],
                )
            else:
                generated_ids = self.sampler.sample(
                    inputs=prompts,
                    config=self.sampler_config,
                    return_dict=False,
                )
                traces = None

            generated_answers = dllm.utils.sample_trim(
                self.tokenizer,
                generated_ids.tolist(),
                prompt_token_lists,
            )

            for answer, gen_kwargs in zip(generated_answers, gen_kwargs_list):
                answer = self._trim_by_stop(answer, gen_kwargs["until"])
                out.append(answer)

            self._append_generation_records(
                batch_start=batch_start,
                contexts=contexts,
                answers=out[-len(batch) :],
                gen_kwargs_list=gen_kwargs_list,
                traces=traces,
                sampler_diagnostics=getattr(
                    self.sampler, "_last_sampler_diagnostics", None
                ),
            )

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
