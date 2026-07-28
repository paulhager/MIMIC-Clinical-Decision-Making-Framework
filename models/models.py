import os
from os.path import join
from typing import Any, List, Mapping, Dict

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from langchain.llms.base import LLM
import tiktoken

from models.hosted_api import HostedChatClient
from agents.agent import STOP_WORDS
from utils.nlp import extract_sections


class CustomLLM(LLM):
    model_name: str
    max_context_length: int
    probabilities: Any = None
    exllama: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    truncation_side: str = "left"
    model: Any
    generator: Any
    tokenizer: Any
    seed: int
    self_consistency: bool = False

    openai_api_key: str = None
    api_key_env: str = None
    api_base_url: str = None
    api_site_url_env: str = None
    api_app_name_env: str = None
    api_provider: Dict[str, Any] = None
    api_reasoning: Dict[str, Any] = None
    max_output_tokens: int = 2048
    hosted_client: Any = None
    last_api_metadata: Dict[str, Any] = None
    tags: Dict[str, str] = None

    @property
    def _llm_type(self) -> Any:
        return "custom"

    @property
    def _llm_name(self) -> str:
        return self.model_name

    @property
    def _llm_device(self) -> str:
        return self.model.device

    @property
    def _llm_8bit(self) -> bool:
        return self.load_in_8bit

    @property
    def _llm_4bit(self) -> bool:
        return self.load_in_4bit

    @property
    def _llm_truncation_side(self) -> str:
        return self.truncation_side

    def load_model(self, base_models: str) -> None:
        if self.model_name == "Human":
            return
        elif self.openai_api_key or self.api_key_env:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Hosted open-weight models are not registered in tiktoken.
                # This is used for conservative prompt-length accounting only.
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            self.hosted_client = HostedChatClient(
                api_key=self.openai_api_key,
                api_key_env=self.api_key_env,
                api_base_url=self.api_base_url,
                site_url_env=self.api_site_url_env,
                app_name_env=self.api_app_name_env,
            )
            return

        import torch

        torch.cuda.empty_cache()

        if (
            self.model_name
            == "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
        ):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k",
                device_map="auto",
                torch_dtype=torch.float16,
            )

        elif "GPTQ" in self.model_name:
            if self.exllama:
                from exllamav2 import ExLlamaV2Cache
                from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
                from models.exllamav2_generator_base_custom import (
                    ExLlamaV2BaseGenerator,
                )

                torch.cuda._lazy_init()
                config = ExLlamaV2Config()
                config.model_dir = join(base_models, self.model_name)
                config.prepare()
                config.max_seq_len = self.max_context_length
                config.scale_pos_emb = 1.0
                config.scale_alpha_value = 1.0
                config.no_flash_attn = False
                self.model = ExLlamaV2(config)
                self.model.load()
                self.tokenizer = ExLlamaV2Tokenizer(config)
                cache = ExLlamaV2Cache(self.model)
                self.generator = ExLlamaV2BaseGenerator(
                    self.model, cache, self.tokenizer
                )
                self.generator.warmup()

            else:
                from auto_gptq import exllama_set_max_input_length
                from transformers import LlamaTokenizer, LlamaForCausalLM

                base_model = join(base_models, self.model_name)

                self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
                self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.model = exllama_set_max_input_length(
                    self.model, self.max_context_length
                )

        elif (
            self.model_name == "meta-llama/Meta-Llama-3-70B-Instruct"
            or self.model_name == "aaditya/OpenBioLLM-Llama3-70B"
            or self.model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct"
            or self.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        ):
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            print(f"loading from {base_models}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=base_models,
            )

            eot = "<|eot_id|>"
            eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id

            print("loaded tokenizer")
            bb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=base_models,
                device_map="auto",
                quantization_config=bb_cfg,
            )
            print("loaded model")

        elif self.model_name == "axiong/PMC_LLaMA_13B":
            from transformers import LlamaTokenizer, LlamaForCausalLM

            self.tokenizer = LlamaTokenizer.from_pretrained("axiong/PMC_LLaMA_13B")
            self.model = LlamaForCausalLM.from_pretrained(
                "axiong/PMC_LLaMA_13B", device_map="auto", torch_dtype=torch.float16
            )

        elif self.model_name == "google/flan-t5-xxl":
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            self.model = T5ForConditionalGeneration.from_pretrained(
                "google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16
            )

        elif self.model_name == "bigscience/T0pp":
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            orig = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

            self.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "bigscience/T0pp", device_map="auto", torch_dtype=torch.float16
            )

            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = orig

        elif self.model_name.startswith("togethercomputer/RedPajama-INCITE"):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, pad_token="[PAD]"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float16, device_map="auto"
            )

        elif self.model_name.startswith("tiiuae/falcon"):
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.load_in_8bit = "40b" in self.model_name
            if torch.cuda.device_count() > 1:
                self.load_in_8bit = False
                device_map = "balanced_low_0"
            else:
                device_map = "sequential"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, pad_token="[PAD]"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=self.load_in_8bit,
                device_map=device_map,
            )

        else:
            raise ValueError("Model name not recognized")

        if not self.model_name.startswith("tiiuae/falcon") and not self.exllama:
            self.model.eval()
            if torch.__version__ >= "2":
                self.model = torch.compile(self.model)

        self.tokenizer.truncation_side = "left"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return self.hosted_client.create(**kwargs)

    def remove_input_tokens(self, output_tokens, ids):
        # Truncate the larger tensor to match the size of the smaller one
        min_size = min(output_tokens.size(1), ids.size(1))
        truncated_output_tokens = output_tokens[:, :min_size]
        truncated_ids = ids[:, :min_size]

        # Element-wise comparison and cumulative product to count length of common prefix
        common_prefix = (
            (truncated_output_tokens == truncated_ids).cumprod(dim=0).sum().item()
        )

        return output_tokens[:, common_prefix:]

    def _call(
        self,
        prompt: str,
        stop: List[str],
        do_sample=True,
        temperature=0.01,
        top_k=1,
        top_p=0.95,
        num_beams=1,
        repetition_penalty=1.2,
        length_penalty=1.0,
        **kwargs,
    ) -> str:
        self.probabilities = None
        effective_stop_words = list(dict.fromkeys([*STOP_WORDS, *(stop or [])]))
        if self.model_name == "Human":
            output = input(prompt)

        elif self.openai_api_key or self.api_key_env:
            messages = extract_sections(
                prompt,
                self.tags,
            )

            request = dict(
                model=self.model_name,
                messages=messages,
                stop=effective_stop_words,
                temperature=0.0,
                seed=self.seed,
                max_tokens=self.max_output_tokens,
            )
            if self.api_provider:
                request["provider"] = self.api_provider
            if self.api_reasoning:
                request["reasoning"] = self.api_reasoning

            response = self.completion_with_backoff(**request)
            self.last_api_metadata = HostedChatClient.response_metadata(response)
            output = response["choices"][0]["message"]["content"]
        elif self.exllama:
            import torch
            from exllamav2.generator import ExLlamaV2Sampler
            from models.utils import create_stop_criteria_exllama

            with torch.inference_mode():
                ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
                tokens_prompt = ids.shape[-1]

                settings = ExLlamaV2Sampler.Settings()
                if self.self_consistency:
                    settings = settings.clone()
                    settings.temperature = 0.7
                    seed = None
                else:
                    settings = settings.greedy_clone()
                    seed = self.seed

                stop_criteria = create_stop_criteria_exllama(
                    effective_stop_words,
                    self.tokenizer.eos_token_id,
                    self.tokenizer,
                )

                output_tokens, self.probabilities = self.generator.generate_simple(
                    prompt,
                    gen_settings=settings,
                    num_tokens=self.max_context_length - tokens_prompt,
                    seed=seed,
                    token_healing=True,
                    encode_special_tokens=True,
                    decode_special_tokens=False,
                    stop_criteria=stop_criteria,
                )

                output_tokens = self.remove_input_tokens(output_tokens, ids)
                output = self.tokenizer.decode(
                    output_tokens, decode_special_tokens=False
                )[0]
        else:
            import torch
            from transformers import GenerationConfig, StoppingCriteriaList
            from models.utils import create_stop_criteria

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_context_length,
                truncation=True,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(self.model.device)

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

            stop_criteria = create_stop_criteria(
                effective_stop_words, self.tokenizer, self.model.device
            )

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([stop_criteria]),
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=self.max_context_length,
                )

            s = generation_output.sequences
            s_no_input = s[:, input_ids.shape[1] :]
            output = self.tokenizer.batch_decode(s_no_input, skip_special_tokens=True)[
                0
            ]

        # Remove observations strings from output if generated
        for stop_word in effective_stop_words:
            output = output.replace(stop_word, "")

        return output.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_name": self.model_name,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
        }
