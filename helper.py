from __future__ import annotations

import logging
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from keys import mykey
from utils.latent_refinement import LatentClinicalThoughtRefiner

# A dictionary to cache models and tokenizers to avoid reloading.
models = {}


def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger:
        if mode == "error":
            logger.error(message)
        elif mode == "warning":
            logger.warning(message)
        else:
            logger.info(message)
    if print_to_std:
        print(message + "\n")


class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        if self.use_api == "openai":
            from openai import OpenAI

            self.api_account = self.args.get("api_account", "openai")
            self.client = OpenAI(api_key=mykey[self.api_account])
        elif self.use_vllm:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch.float16
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._ensure_tokenizer_tokens()
            except Exception as e:
                log_info(
                    f"[ERROR] [{self.model_name}]: VLLM fallback to HuggingFace due to error: {str(e)}",
                    mode="error",
                )
                self.use_vllm = False

        if not self.use_vllm and self.use_api != "openai":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            self._ensure_tokenizer_tokens()
            log_info(f"[INFO] [{self.model_name}] loaded on device: {self.device}")

    def _ensure_tokenizer_tokens(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        terminators = [self.tokenizer.eos_token_id]
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in terminators:
            terminators.append(eot_id)
        self.terminators = terminators

    def generate(self, messages: List[Dict[str, str]], **kwargs):
        triplet_prefix = kwargs.pop("triplet_prefix", None)
        if kwargs:
            self.args.update({k: v for k, v in kwargs.items() if v is not None})

        log_info(f"[{self.model_name}][INPUT]: {messages}")
        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", self.args.get("max_length", 256))
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if triplet_prefix is not None and self.use_api != "openai" and self.model is not None:
            return self.graphmed_generate(messages, triplet_prefix)
        if self.use_api == "openai":
            return self.openai_generate(messages)
        if self.use_vllm:
            return self.vllm_generate(messages)
        return self.huggingface_generate(messages)

    def _input_ids_from_messages(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            log_info(f"[{self.model_name}]: Could not apply chat template.", mode="warning")
            prompt = "\n\n".join([m["content"] for m in messages])
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self.device)

    def huggingface_generate(self, messages):
        inputs = self._input_ids_from_messages(messages)
        outputs = self.model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=int(self.max_tokens),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.terminators,
        )
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        usage = {"input_tokens": inputs.shape[-1], "output_tokens": outputs.shape[-1] - inputs.shape[-1]}
        output_dict = {"response_text": response_text, "usage": usage}
        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage

    def _sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature is None or float(self.temperature) <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / float(self.temperature)
        probs = torch.softmax(logits, dim=-1)
        top_p = float(self.top_p) if self.top_p is not None else 1.0
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return sorted_indices.gather(-1, sampled)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def graphmed_generate(self, messages, triplet_prefix):
        if not torch.is_tensor(triplet_prefix) or triplet_prefix.numel() == 0:
            return self.huggingface_generate(messages)

        input_ids = self._input_ids_from_messages(messages)
        model_dtype = next(self.model.parameters()).dtype
        evidence_tokens = triplet_prefix.to(device=self.device, dtype=model_dtype)
        if evidence_tokens.ndim == 2:
            evidence_tokens = evidence_tokens.unsqueeze(0)

        refinement_steps = int(self.args.get("refinement_steps", 5))
        refiner = LatentClinicalThoughtRefiner(self.model, refinement_steps=refinement_steps)
        refined_context = refiner(evidence_tokens)
        prompt_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([refined_context, prompt_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        generated = []
        eos_ids = set(i for i in (self.terminators or []) if isinstance(i, int) and i >= 0)
        for _ in range(int(self.max_tokens)):
            next_token = self._sample_next_token(next_logits)
            token_id = int(next_token.item())
            if token_id in eos_ids:
                break
            generated.append(token_id)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=torch.long, device=self.device)],
                dim=1,
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        response_text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        usage = {
            "input_tokens": int(inputs_embeds.shape[1]),
            "output_tokens": len(generated),
        }
        output_dict = {"response_text": response_text, "usage": usage}
        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage

    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            log_info(f"[{self.model_name}]: Could not apply chat template.", mode="warning")
            inputs = "\n\n".join([m["content"] for m in messages])

        from vllm import SamplingParams

        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=int(self.max_tokens),
            top_p=self.top_p,
            logprobs=self.top_logprobs,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
        logprobs = outputs[0].outputs[0].cumulative_logprob
        usage = {
            "input_tokens": len(outputs[0].prompt_token_ids),
            "output_tokens": len(outputs[0].outputs[0].token_ids),
        }
        output_dict = {"response_text": response_text, "usage": usage}
        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        if self.top_logprobs == 0:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=int(self.max_tokens),
                top_p=self.top_p,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=int(self.max_tokens),
                top_p=self.top_p,
                logprobs=True,
                top_logprobs=self.top_logprobs,
            )

        num_input_tokens = response.usage.prompt_tokens
        num_output_tokens = response.usage.completion_tokens
        response_text = response.choices[0].message.content.strip()
        log_probs = response.choices[0].logprobs if self.top_logprobs > 0 else None
        log_info(f"[{self.model_name}][OUTPUT]: {response}")
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    if "gpt" in model_name or "o1" in model_name:
        use_api = "openai"

    init_kwargs = {k: v for k, v in kwargs.items() if k != "triplet_prefix"}
    model_cache = models.get(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **init_kwargs)
        models[model_name] = model_cache

    return model_cache.generate(messages, **kwargs)
