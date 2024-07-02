from typing import List
import torch
from transformers import StoppingCriteria


def calculate_prob_confidence(probs: torch.Tensor):
    return torch.pow(torch.prod(probs), 1 / len(probs))


def calculate_log_prob_confidence(probs: torch.Tensor):
    # diagnosis_idx = prediction.index(diagnosis)
    # diagnosis_probs = probs[diagnosis_idx : diagnosis_idx + len(diagnosis)]
    log_probs = torch.log(probs)
    sum_log_probs = torch.sum(log_probs)
    avg_log_prob = sum_log_probs / len(probs)
    return avg_log_prob


def create_stop_criteria(stop_words: List[str], tokenizer, device) -> StoppingCriteria:
    stop_ids = [
        tokenizer.encode(w, add_special_tokens=False, return_tensors="pt").to(device)
        for w in stop_words
    ]
    # When a word occurs within a continuous string its encoding changes. To account for this, we add a period to the beginning of the word and remove it from encoding.
    stop_ids_2 = [
        tokenizer.encode("." + w, add_special_tokens=False, return_tensors="pt")[
            :, 1:
        ].to(device)
        for w in stop_words
    ]
    stop_ids.extend(stop_ids_2)
    stop_ids = [s[0] for s in stop_ids]
    return KeywordsStoppingCriteria(stop_ids)


def create_stop_criteria_exllama(
    stop_words: List[str], stop_token: int, tokenizer
) -> StoppingCriteria:
    stop_ids = [tokenizer.encode(w) for w in stop_words]
    # When a word occurs within a continuous string its encoding changes. To account for this, we add a period to the beginning of the word and remove it from encoding.
    stop_ids_2 = [tokenizer.encode("." + w)[:, 1:] for w in stop_words]
    stop_ids.extend(stop_ids_2)
    stop_ids = [s[0] for s in stop_ids]
    stop_ids.append(torch.tensor([stop_token]))
    return KeywordsStoppingCriteria(stop_ids)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords: torch.Tensor):
        self.keywords = keywords

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for k in self.keywords:
            if torch.equal(input_ids[0][-len(k) :], k):
                return True
        return False
