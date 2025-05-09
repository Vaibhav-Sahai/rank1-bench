import argparse
import mteb
from mteb import MTEB
import logging
import os
import json

from functools import partial
import logging
import math
from typing import Any, Callable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch
from mteb.model_meta import ModelMeta
from mteb.models.rerankers_custom import RerankerWrapper
import random

# prompts for sys

SYSTEM_PROMPTS = [
    "You are a helpful assistant. You are going to be given a query and a document. Given the query and the document, please think, and then determine if the document is relevant to the query.",
    
    "As an AI trained to evaluate information relevance, analyze the provided query and document. Determine whether the document contains information that directly answers or is closely related to the query.",
    
    "You are a relevance assessment expert. Examine the relationship between the given query and document. Evaluate whether the document provides useful information that addresses the query's intent.",
    
    "Acting as a search quality evaluator, review the query and document pair. Assess if the document would satisfy a user who submitted this query based on its relevance and information value.",
    
    "You are a precision-focused relevance judge. For the provided query and document, determine if there is a meaningful connection that would make this document a valuable result for someone searching with this query.",
    
    "As an information retrieval specialist, analyze whether the given document contains content that directly addresses, explains, or provides context for the query. Determine if it's relevant.",
    
    "You are a semantic matching expert. Evaluate whether the concepts, topics, and information in the document align with the information need expressed in the query.",
    
    "Acting as a relevance classifier, determine if the document contains information that would help answer the query. Consider both explicit and implicit relevance signals.",
    
    "You are a query-document relevance analyzer. Assess whether the document provides information that satisfies the intent behind the query, even if it doesn't contain the exact query terms.",
    
    "As a search relevance evaluator, determine if this document would be a helpful result for the given query. Consider factors like topical relevance, information completeness, and query intent.",
    
    "You are an information need assessor. Given a user query and a document, evaluate whether the document contains information that would fulfill the user's information need expressed in the query.",
    
    "Acting as a relevance determination system, analyze the semantic relationship between the query and document. Decide if the document contains information that addresses what the user is looking for.",
    
    "You are a search quality rater. Evaluate whether the document provides valuable information related to the query that would satisfy a user's search intent.",
    
    "As a content relevance expert, determine if the document contains information that directly or indirectly addresses the information need expressed in the query.",
    
    "You are a query-document matcher. Assess whether the document's content is semantically related to and addresses the information need in the query.",
    
    "Acting as a relevance judgment system, evaluate whether the document would be considered useful by a user who submitted the given query.",
    
    "You are a search result evaluator. Determine if the document contains information that would make it a good search result for the given query.",
    
    "As an information relevance detector, analyze whether the document contains content that directly addresses or is closely related to the topic of the query.",
    
    "You are a query intent analyzer. Given a query and document, determine if the document satisfies the likely intent behind the query.",
    
    "Acting as a relevance assessment tool, evaluate whether the document provides information that would be useful to someone searching for the given query.",
    
    "You are a document relevance classifier. Analyze the query and document to determine if there is sufficient topical and semantic overlap to consider the document relevant to the query.",
    
    "As a search relevance judge, determine if the document contains information that directly addresses or provides valuable context for understanding the topic of the query.",
    
    "You are an information matching expert. Evaluate whether the document would satisfy the information need expressed in the query based on content relevance and usefulness.",
    
    "Acting as a relevance evaluation system, determine if the document contains information that would be considered valuable by someone looking for an answer to the given query."
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class rank1ft(RerankerWrapper):
    name: str = "rank1ft"

    def __init__(
        self,
        model_name_or_path: str = "jhu-clsp/rank1-7b",
        batch_size: int = 999999999999,
        context_size: int = 8196,
        max_output_tokens: int = 1024,
        fp_options: str = "float32",
        num_gpus: int = 1,
        device: str = "cuda",
        force_rethink: int = 0,
        dataset_prompt: str = None,
        **kwargs,
    ):
        """
        rank1 is a reasoning reranker model (using test-time compute) which generates a reasoning chain before deciding true or false

        Args:
            model_name_or_path: Path to the model or name of the model on HuggingFace Hub
            batch_size: Maximum batch size for processing (default: very large number to let vLLM handle batching)
            context_size: Maximum context length for the model (default: 4096)
            max_output_tokens: Maximum number of tokens to generate (default: 1024)
            fp_options: Floating point precision to use, e.g. 'float16' (default: 'float16')
            num_gpus: Number of GPUs to use for tensor parallelism (default: 1)
            device: Device to load the model on (default: 'cuda')
            force_rethink: Number of times to force model to rethink its answer (default: 0)
            **kwargs: Additional keyword arguments passed to parent RerankerWrapper
        """
        super().__init__(model_name_or_path, batch_size=batch_size, fp_options=fp_options, **kwargs)
        
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self.num_gpus = num_gpus
        self.device = device
        self.force_rethink = force_rethink
        self.model_name_or_path = model_name_or_path
        self.dataset_prompt = dataset_prompt

        # Initialize tokenizer with max length of 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache commonly used token IDs
        self.true_token = self.tokenizer("True", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("False", add_special_tokens=False).input_ids[0]
        self.think_token = self.tokenizer("<relevance>", add_special_tokens=False).input_ids[0]
        self.think_end_token = self.tokenizer("</relevance>", add_special_tokens=False).input_ids[-1]

        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=0.98,
            dtype=fp_options,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_output_tokens,
            logprobs=20,
            stop=["<relevance>"],
            skip_special_tokens=False
        )

    def _fix_incomplete_responses(
        self, 
        original_prompts: List[str], 
        generated_texts: List[str]
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        This function is used to fix incomplete responses from the vLLM model. In some cases the model does not generate the end </think> token.
            In these cases, we should force it to generate it so that we have some prediction. 

        Args:
            original_prompts: The original prompts that were used to generate the texts
            generated_texts: The texts that were generated by the vLLM model

        Returns:
            final_texts: The texts that were generated by the vLLM model + the outputs from the forcing step
            token_counts: The number of tokens in the texts total
            scores: The scores of the texts
        """             
        cleaned_texts = []
        for text in generated_texts:
            text = text.rstrip()
            if not text.endswith(('.', '!', '?')):
                last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                if last_punct != -1:
                    text = text[:last_punct + 1]
            cleaned_texts.append(text.strip())
        
        forced_prompts = [
            f"{original_prompt}\n{cleaned_text}\n<relevance>" 
            for original_prompt, cleaned_text in zip(original_prompts, cleaned_texts)
        ]

        new_sampling_args = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
            skip_special_tokens=False
        )
        outputs = self.model.generate(forced_prompts, new_sampling_args)

        # get the next token logits of just the next token
        all_final_texts = []
        all_token_counts = []
        all_scores = []    
        for i in range(len(outputs)):
            try:
                text = outputs[i].outputs[0].text
                final_logits = outputs[i].outputs[0].logprobs[-1]
                assert self.false_token in final_logits and self.true_token in final_logits, f"final logits are missing true or false: {final_logits}"
            except Exception as e:
                print(f"Error: {e} on fixing error, setting at 0.5 score: {outputs[i].outputs}")
                all_scores.append(0.5)
                all_token_counts.append(len(outputs[i].outputs[0].token_ids))
                all_final_texts.append(text)
                continue
                
            token_count = len(outputs[i].outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            
            all_final_texts.append(text)
            all_token_counts.append(token_count)
            all_scores.append(score)
        
        return all_final_texts, all_token_counts, all_scores

    def truncate_input(self, text: str) -> str:
        """
        Truncate the input text to the context size. This is not used, except if you are using the Llama 8B quantized model
        """
        if len(self.tokenizer(text)["input_ids"]) >= self.context_size:
            return self.tokenizer.decode(self.tokenizer(text)["input_ids"][:self.context_size])
        else:
            return text

    def _process_with_vllm(self, prompts):
        """
        vLLM is significantly faster than HF, so we use it by default. This function handles the cases where the model does not generate the end </think> token.

        Args:
            prompts: The prompts to generate from

        Returns:
            outputs: The outputs from the vLLM model
        """
        prompts = [self.truncate_input(prompt) for prompt in prompts]
        outputs = self.model.generate(prompts, self.sampling_params)

        # Pre-allocate lists with None values
        total_length = len(prompts)
        all_outputs = [None] * total_length
        all_output_token_counts = [None] * total_length
        all_scores = [None] * total_length
        
        incomplete_prompts = []
        incomplete_texts = []
        incomplete_indices = []
        
        # Process complete responses first
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            print(f"DEBUG - Text: {text}")
            try:
                final_logits = output.outputs[0].logprobs[-1]
            except Exception as e:
                print(f"Error: {e} on getting final logits: {output.outputs[0]}")
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue
            
            print(f"DEBUG: bool check {self.true_token in final_logits} {self.false_token in final_logits}")
            if self.true_token not in final_logits or self.false_token not in final_logits:
                incomplete_prompts.append(prompts[i])
                incomplete_texts.append(text)
                incomplete_indices.append(i)
                continue
                
            token_count = len(output.outputs[0].token_ids)
            true_logit = final_logits[self.true_token].logprob
            false_logit = final_logits[self.false_token].logprob
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            
            all_outputs[i] = text
            all_output_token_counts[i] = token_count
            all_scores[i] = score
            
            # print(f"DEBUG: {true_logit} {false_logit} {true_score} {false_score} {score}")
            # print(f"DEBUG - Score: {score}")
        
        print(f"DEBUG: {len(incomplete_prompts)} incomplete prompts")
        # Handle incomplete responses
        if incomplete_indices:
            fixed_texts, fixed_counts, fixed_scores = self._fix_incomplete_responses(
                incomplete_prompts, incomplete_texts
            )
            
            # Fill in the fixed responses at their original positions
            for orig_idx, (text, count, score) in zip(
                incomplete_indices, zip(fixed_texts, fixed_counts, fixed_scores)
            ):
                all_outputs[orig_idx] = text
                all_output_token_counts[orig_idx] = count
                all_scores[orig_idx] = score

        return all_outputs, all_output_token_counts, all_scores

    def return_prompt(self, query, doc_content, prompt) -> str:
        query = prompt.replace("FILL_QUERY_HERE", query) if prompt else query
        system_prompt = random.choice(SYSTEM_PROMPTS)
        return f"""{system_prompt}

<query> 
{query} 
</query>

<document>
{doc_content} 
</document>
Enclose your final answer as True or False in <relevance> tags. Enclose your thinking in <think> tags.
<think>"""

    def _prepare_prompts_for_rethink(self, prompts: List[str], texts: List[str], rethink_text: str = "Wait") -> List[str]:
        """Prepare prompts for the rethinking step."""
        full_texts = [p + t for p, t in zip(prompts, texts)]
        stripped_texts = [t.split("</think>")[0] for t in full_texts]
        just_generated_texts = [t.split("</think>")[0] for t in full_texts]
        return [s + f"\n{rethink_text}" for s in stripped_texts], just_generated_texts

    @torch.inference_mode()
    def predict(self, input_to_rerank, **kwargs):
        """This is setup to run with mteb but can be adapted to your purpose"""
        inputs = list(zip(*input_to_rerank))
        if len(input_to_rerank[0]) == 2:
            queries, passages = inputs
            instructions = None
        else:
            queries, passages, instructions = inputs

        # queries = queries[:10]
        
        if instructions is not None and instructions[0] is not None:
            queries = [f"{q} {i}".strip() if q.strip() != i.strip() else q.strip() for i, q in zip(instructions, queries)]

        if isinstance(passages[0], dict):
            passages = [f"{v['title']} {v['text']}" if 'title' in v else v['text'] for v in passages]

        prompts = [
            self.return_prompt(query, passage, self.dataset_prompt)
            for query, passage in zip(queries, passages)
        ]
        print(f"Example prompt: ```\n{prompts[0]}\n```")

        texts, token_counts, scores = self._process_with_vllm(prompts)
        while self.force_rethink:
            revised_prompts, previously_generated_texts = self._prepare_prompts_for_rethink(prompts, texts)
            new_texts, new_token_counts, new_scores = self._process_with_vllm(revised_prompts)
            # add to the previous output
            texts = [prev + f"\n{rethink_text}" + f"{new_text}" for prev, new_text in zip(texts, new_texts)]
            scores = new_scores
            token_counts = [prev_token_count + new_token_count for prev_token_count, new_token_count in zip(token_counts, new_token_counts)]
            self.force_rethink -= 1

        self.response_data = {
            "scores": scores,
            "texts": texts,
            "token_counts": token_counts,
            "prompts": prompts,
        }
        
        return scores