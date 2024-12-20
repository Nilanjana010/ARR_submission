# ARR_submission
This repo consists of two folders:-
1) Situation-Driven-Adversarial-Attacks-main:-
    -> The codes to attack the different LLMs: gpt-3.5-turbo-0125, phi-1.5, gpt-4, gemma-7b, 
    -> Meta-Llama-3-8B, and the 4-bit quantized Llama-2 7B chat.
    -> [GPT-4 Judge](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) code that we used in our research.
    -> llama.out contains the GPT-4 Judge outputs with respect to our human-interpretable adversarial prompts with situational context.

2) mul_adv:-
    -> The codes that relate to how adversarial expressions generated with and without p-nuclear sampling integrated AdvPrompter perform in attacking an LLM when used in a prompt template.
    -> The codes without any adversarial insertion.
    -> Codes involve attacking several models.
    -> GPT-4 Judge is now used as gpt-4o-mini
