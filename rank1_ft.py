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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class rank1ft(RerankerWrapper):
    name: str = "rank1ft"

    def __init__(
        self,
        model_name_or_path: str = "jhu-clsp/rank1-7b",
        batch_size: int = 999999999999,
        context_size: int = 16000,
        max_output_tokens: int = 1024,
        fp_options: str = "float16",
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
        self.true_token = self.tokenizer("1", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("0", add_special_tokens=False).input_ids[0]
        self.think_token = self.tokenizer("<relevance>", add_special_tokens=False).input_ids[0]
        self.think_end_token = self.tokenizer("</relevance>", add_special_tokens=False).input_ids[-1]

        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=0.95,
            dtype=fp_options,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_output_tokens,
            logprobs=20,
            stop=["<relevance>1", "<relevance>0"],
            skip_special_tokens=False
        )

    def _fix_incomplete_responses(
        self, 
        original_prompts: List[str], 
        generated_texts: List[str]
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        This function is used to fix incomplete responses from the vLLM model. In some cases the model does not generate the end </reasoning> token.
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
        vLLM is significantly faster than HF, so we use it by default. This function handles the cases where the model does not generate the end </reasoning> token.

        Args:
            prompts: The prompts to generate from

        Returns:
            outputs: The outputs from the vLLM model
        """
        # prompts = [self.truncate_input(prompt) for prompt in prompts]
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
            
            print(f"DEBUG: {true_logit} {false_logit} {true_score} {false_score} {score}")
            print(f"DEBUG - Score: {score}")
        
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
        return f"""
You are tasked with determining whether a given document is relevant to answering a specific query. Follow these steps carefully:
1. You will be presented with a query and a document containing a title and content.
2. First, examine the query.
3. Next, examine the document.
4. Analyze the query carefully. Determine what specific information or answer it is seeking. Consider the key concepts, entities, or relationships mentioned in the query.
5. Carefully read and analyze the document content. Pay attention to the main topics, key information, and any details that might be relevant to the query.
6. Reason about the relevance of the document to the query. Consider the following:
- Does the document contain information that directly answers the query?
- Does it provide context or background information that would be helpful in understanding or answering the query?
- Are there any significant matches between key terms or concepts in the query and the document?
- Even if the document doesn't fully answer the query, does it contain partial information that could contribute to an answer?
7. Based on your analysis, determine whether the document is relevant or not relevant to answering the query.
8. Provide your reasoning and verdict in the following format:
<reasoning> [Explain your thought process here, discussing why you believe the document is or is not relevant to the query. Provide specific examples or quotes from the document if applicable.] </reasoning>
<relevance>[Insert either 0 for not relevant or 1 for relevant]</relevance>
Remember, the content within the <relevance> tags must be either 0 or 1, with no other text or explanation.

<question> 
{query} 
</question>

<document>
{doc_content} 
</document>
<reasoning>"""

    def _prepare_prompts_for_rethink(self, prompts: List[str], texts: List[str], rethink_text: str = "Wait") -> List[str]:
        """Prepare prompts for the rethinking step."""
        full_texts = [p + t for p, t in zip(prompts, texts)]
        stripped_texts = [t.split("</reasoning>")[0] for t in full_texts]
        just_generated_texts = [t.split("</reasoning>")[0] for t in full_texts]
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

        return scores