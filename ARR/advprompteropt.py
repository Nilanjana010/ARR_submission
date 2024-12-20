# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from tqdm import tqdm

from sequence import MergedSeq, Seq, EmptySeq
from utils import apply_repetition_penalty


@torch.no_grad()
def advPrompterOpt(cfg, instruct, target, prompter, target_llm):
    #if cfg.verbose:
    tqdm.write("\n Running AdvPrompterOpt: Generating optimized suffix...")

    # Compute the initial prediction losses without a suffix
    full_instruct = Seq(
        text=instruct.text, tokenizer=target_llm.tokenizer, device=target_llm.device
    )
    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
    )
    losses = target_llm_tf.loss_batch.detach().to(prompter.device)  # (ebs, )
    tqdm.write(f" losses in advPrompterOpt: {losses}")

    # Initialize the beam scores
    beam_scores = torch.zeros_like(losses)  # (ebs, )
    suffix_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)
    tqdm.write(f"beam_scores in advPrompterOpt: {beam_scores}")
    tqdm.write(f"suffix_beams in advPrompterOpt: {suffix_beams}")

    for idx in range(cfg.train.q_params.max_new_tokens):
        if idx == 0:
            num_beams_in = 1
            num_beams_out = cfg.train.q_params.num_beams
        elif idx == cfg.train.q_params.max_new_tokens - 1:
            num_beams_in = cfg.train.q_params.num_beams
            num_beams_out = 1
        else:
            num_beams_in = cfg.train.q_params.num_beams
            num_beams_out = cfg.train.q_params.num_beams

        # expand the dimension of instruct and targets to match suffix beams
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)
        tqdm.write(f"instruct_rep in advPrompterOpt: {instruct_rep}")
        tqdm.write(f"target_rep in advPrompterOpt: {target_rep}")

        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            cfg=cfg, instruct=instruct_rep, suffix=suffix_beams, prompter=prompter
        )
        tqdm.write(f"next_dist_seq_prompter in advPrompterOpt: {next_dist_seq_prompter}")
        tqdm.write(f"next_dist_seq_basemodel in advPrompterOpt: {next_dist_seq_basemodel}")

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                cfg=cfg,
                instruct=instruct_rep,
                target=target_rep,
                suffix=suffix_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
            )
        )
        tqdm.write(f"next_token_candidate_ids in advPrompterOpt: {next_token_candidate_ids}")
        tqdm.write(f"candidate_beam_scores in advPrompterOpt: {candidate_beam_scores}")
        tqdm.write(f"candidate_losses in advPrompterOpt: {candidate_losses}")

        suffix_beams, losses, beam_scores = select_next_beams(
            cfg=cfg,
            suffix_beams=suffix_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
        )
        tqdm.write(f"suffix_beams in advPrompterOpt: {suffix_beams}")
        tqdm.write(f"losses in advPrompterOpt: {losses}")
        tqdm.write(f"beam_scores in advPrompterOpt: {beam_scores}")

        #if cfg.verbose:
        tqdm.write(f" Beams[0] (iter {idx}): {suffix_beams[:num_beams_out].text}")

    #if cfg.verbose:
    tqdm.write(
            f" AdvPrompterOpt completed. Generated suffix[0]: {suffix_beams[0].text}"
        )
    return suffix_beams


def get_next_token_probabilities(cfg, instruct, suffix, prompter):
    # get the next token probabilities from the prompter and the base model
    prompter_next = prompter.get_next_token(
        key="suffix", instruct=instruct, suffix=suffix
    )
    tqdm.write(f"prompter_next in advPrompterOpt: {prompter_next}")

    next_dist_seq_prompter = (
        prompter_next.response_dist.clone().detach()
    )  # (ebs, 1, vocab_size)
    tqdm.write(f"next_dist_seq_prompter response_dist in advPrompterOpt: {next_dist_seq_prompter.logits}")
    tqdm.write(f"next_dist_seq_prompter response_dist shape in advPrompterOpt: {next_dist_seq_prompter.logits.shape}")

    prompter_next_basemodel = prompter.get_next_token(
        key="suffix", instruct=instruct, suffix=suffix, use_basemodel=True
    )
    tqdm.write(f"prompter_next_basemodel in advPrompterOpt: {prompter_next_basemodel}")

    next_dist_seq_basemodel = (
        prompter_next_basemodel.response_dist.clone().detach()
    )  # (bs, 1, vocab_size)
    tqdm.write(f"next_dist_seq_basemodel response_dist in advPrompterOpt: {next_dist_seq_basemodel.logits}")
    tqdm.write(f"next_dist_seq_basemodel response_dist shape in advPrompterOpt: {next_dist_seq_basemodel.logits.shape}")

    # apply repetition penalty
    if not suffix.is_empty and "repetition_penalty" in cfg.train.q_params:
        next_dist_logits_basemodel = apply_repetition_penalty(
            logits=next_dist_seq_basemodel.logits.squeeze(1),
            prev_ids=suffix.ids,
            penalty=cfg.train.q_params.repetition_penalty,
        )
        tqdm.write(f"next_dist_logits_basemodel in advPrompterOpt: {next_dist_logits_basemodel}")
        tqdm.write(f"next_dist_logits_basemodel shape in advPrompterOpt: {next_dist_logits_basemodel.shape}")

        next_dist_seq_basemodel = Seq(
            logits=next_dist_logits_basemodel[:, None, :],
            mask=next_dist_seq_basemodel.mask,
            tokenizer=next_dist_seq_basemodel.tokenizer,
            device=next_dist_seq_basemodel.device,
        )
        tqdm.write(f"next_dist_seq_basemodel in advPrompterOpt: {next_dist_seq_basemodel.logits}")
        tqdm.write(f"next_dist_seq_basemodel shape in advPrompterOpt: {next_dist_seq_basemodel.logits.shape}")


        next_dist_logits_prompter = apply_repetition_penalty(
            logits=next_dist_seq_prompter.logits.squeeze(1),
            prev_ids=suffix.ids,
            penalty=cfg.train.q_params.repetition_penalty,
        )
        tqdm.write(f"next_dist_logits_prompter in advPrompterOpt: {next_dist_logits_prompter}")
        tqdm.write(f"next_dist_logits_prompter shape in advPrompterOpt: {next_dist_logits_prompter.shape}")


        next_dist_seq_prompter = Seq(
            logits=next_dist_logits_prompter[:, None, :],
            mask=next_dist_seq_prompter.mask,
            tokenizer=next_dist_seq_prompter.tokenizer,
            device=next_dist_seq_prompter.device,
        )
        tqdm.write(f"next_dist_seq_prompter in advPrompterOpt: {next_dist_seq_prompter.logits}")
        tqdm.write(f"next_dist_seq_prompter shape in advPrompterOpt: {next_dist_seq_prompter.logits.shape}")

    return next_dist_seq_prompter, next_dist_seq_basemodel


def select_and_evaluate_next_token_candidates(
    cfg,
    instruct,
    target,
    suffix,
    target_llm,
    next_dist_seq_prompter,
    next_dist_seq_basemodel,
    beam_scores,
    prev_losses,
    num_beams_in,
):
    num_chunks = cfg.train.q_params.num_chunks
    assert cfg.train.q_params.top_k % (num_chunks * num_beams_in) == 0
    num_samples_per_beam = cfg.train.q_params.top_k // (num_chunks * num_beams_in)
    tqdm.write(f"num_samples_per_beam in select_and_evaluate_next_token_candidates: {num_samples_per_beam}")
    all_next_token_candidate_ids = None

    for i in range(cfg.train.q_params.num_chunks):
        next_token_candidate_ids = select_next_token_candidates(
            cfg=cfg,
            next_dist_seq=next_dist_seq_prompter,
            previous_next_token_candidate_ids=all_next_token_candidate_ids,
            num_samples_per_beam=num_samples_per_beam,
            always_include_best=cfg.train.q_params.candidates.always_include_best
            and i == 0,
        )  # (ebs = bs * num_beams_in, num_samples_per_beam)

        candidate_beam_scores, candidate_losses = evaluate_next_token_candidates(
            cfg=cfg,
            instruct=instruct,
            target=target,
            suffix=suffix,
            target_llm=target_llm,
            next_token_candidate_ids=next_token_candidate_ids,
            next_dist_seq_basemodel=next_dist_seq_basemodel,
            next_dist_seq_prompter=next_dist_seq_prompter,
            prev_beam_scores=beam_scores,
            prev_losses=prev_losses,
        )  # (ebs, num_samples_per_beam)

        if all_next_token_candidate_ids is None:
            all_next_token_candidate_ids = next_token_candidate_ids
            all_candidate_beam_scores = candidate_beam_scores
            all_candidate_losses = candidate_losses
        else:
            all_next_token_candidate_ids = torch.cat(
                (next_token_candidate_ids, all_next_token_candidate_ids), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_beam_scores = torch.cat(
                (candidate_beam_scores, all_candidate_beam_scores), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_losses = torch.cat(
                (candidate_losses, all_candidate_losses), dim=1
            )  # (ebs, i * num_samples_per_beam)
    return all_next_token_candidate_ids, all_candidate_beam_scores, all_candidate_losses


@torch.no_grad()
def select_next_token_candidates(
    cfg,
    next_dist_seq,
    previous_next_token_candidate_ids,
    num_samples_per_beam,
    always_include_best,
):

    # clone is important here! We modify the logits but will also use the original dist
    next_dist_logits = next_dist_seq.logits.squeeze(1).clone()  # (ebs, vocab_size)
    sh1, sh2 = next_dist_logits.shape
    tqdm.write(f"next_dist_logits in select_next_token_candidates1: {next_dist_logits}")
    tqdm.write(f"previous_next_token_candidate_ids in select_next_token_candidates: {previous_next_token_candidate_ids}")
    tqdm.write(f"next_dist_logits shapeeeee in select_next_token_candidates1: {next_dist_logits.shape}")
    #tqdm.write(f"previous_next_token_candidate_ids shapeeee in select_next_token_candidates: {previous_next_token_candidate_ids.shape}")

    if previous_next_token_candidate_ids is not None:
        previous_next_token_candidate_ids_khot = torch.scatter(
            torch.zeros_like(next_dist_logits), 1, previous_next_token_candidate_ids, 1
        )  # (ebs, vocab_size)
        next_dist_logits -= 1e10 * previous_next_token_candidate_ids_khot
    tqdm.write(f"next_dist_logits in select_next_token_candidates2: {next_dist_logits}")
    tqdm.write(f"next_dist_logits shapeeeeee in select_next_token_candidates29999: {next_dist_logits.shape}")

    if cfg.train.q_params.candidates.do_sample:
        next_dist_logits_new = torch.zeros_like(next_dist_logits[0, :])

        if always_include_best:
            next_dist_logits -= 1e10 * next_dist_seq.onehot.squeeze(1)
        tqdm.write(f"next_dist_logits in select_next_token_candidates3: {next_dist_logits}")
        tqdm.write(f"next_dist_logits shapeeee in select_next_token_candidates300: {next_dist_logits.shape}")
       
        # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
        for i in range(next_dist_logits.shape[0]):
            logits = next_dist_logits[i, :]
            logits = logits / cfg.train.q_params.temperature_new
            filtered_logits = top_k_top_p_filtering(logits, top_k=cfg.train.q_params.top_k_new, top_p=cfg.train.q_params.top_p_new)
            filtered_logits = filtered_logits.clone()
            if i == 0:
                next_dist_logits_new = filtered_logits
            else:
                next_dist_logits_new = torch.cat((next_dist_logits_new, filtered_logits))
          
        tqdm.write(f"next_dist_logits_new.shape: {next_dist_logits_new.shape}")
        next_dist_logits = next_dist_logits_new.clone()
        next_dist_logits = next_dist_logits.reshape(
        sh1, sh2
        )

        probs = torch.softmax(
            next_dist_logits / cfg.train.q_params.candidates.temperature,
            dim=-1,
        )  # (ebs, vocab_size)
        tqdm.write(f"probs in select_next_token_candidates3: {probs}")
        tqdm.write(f"probs shapeeeee in select_next_token_candidates399999: {probs.shape}")

        next_token_candidate_ids = probs.multinomial(
            num_samples=num_samples_per_beam, replacement=False
        )  # (ebs, num_samples_per_beam)
        tqdm.write(f"next_token_candidate_ids in select_next_token_candidates33: {next_token_candidate_ids}")
        tqdm.write(f"next_token_candidate_ids shape in select_next_token_candidates33: {next_token_candidate_ids.shape}")


        if always_include_best:
            next_token_candidate_ids = torch.cat(
                [next_dist_seq.ids, next_token_candidate_ids[:, :-1]], dim=1
            )
        tqdm.write(f"next_token_candidate_ids in select_next_token_candidates44: {next_token_candidate_ids}")
        tqdm.write(f"next_token_candidate_ids shape in select_next_token_candidates33: {next_token_candidate_ids.shape}")


    else:
        next_token_candidate_ids = next_dist_logits.topk(
            k=num_samples_per_beam, dim=-1
        ).indices  # (ebs, num_samples_per_beam)
    return next_token_candidate_ids


@torch.no_grad()
def evaluate_next_token_candidates(
    cfg,
    instruct,
    target,
    suffix,
    target_llm,
    next_token_candidate_ids,
    next_dist_seq_basemodel,
    next_dist_seq_prompter,
    prev_beam_scores,
    prev_losses,
):
    ebs, num_samples_per_beam = next_token_candidate_ids.shape
    tqdm.write(f"num_samples_per_beam in evaluate_next_token_candidates: {num_samples_per_beam}")
    tqdm.write(f"ebs in evaluate_next_token_candidates: {ebs}")

    q_next_token_candidate_ids = torch.reshape(
        next_token_candidate_ids, (ebs * num_samples_per_beam, 1)
    )
    tqdm.write(f"q_next_token_candidate_ids in evaluate_next_token_candidates: {q_next_token_candidate_ids}")
    tqdm.write(f"q_next_token_candidate_ids shape in evaluate_next_token_candidates: {q_next_token_candidate_ids.shape}")

    q_sample_seq = Seq(
        ids=q_next_token_candidate_ids,
        tokenizer=next_dist_seq_prompter.tokenizer,
        device=next_dist_seq_prompter.device,
    )
    tqdm.write(f"q_sample_seq in evaluate_next_token_candidates: {q_sample_seq}")
    # tqdm.write(f"q_sample_seq shape in evaluate_next_token_candidates: {q_sample_seq.shape}")

    # extend to match the extended batch size
    instruct_rep = instruct.repeat_interleave(num_samples_per_beam, dim=0)
    target_rep = target.repeat_interleave(num_samples_per_beam, dim=0)
    tqdm.write(f"instruct_rep in evaluate_next_token_candidates: {instruct_rep}")
    tqdm.write(f"target_rep in evaluate_next_token_candidates: {target_rep}")
    tqdm.write(f"instruct_rep text in evaluate_next_token_candidates: {instruct_rep.text}")
    tqdm.write(f"target_rep text in evaluate_next_token_candidates: {target_rep.text}")
    if not suffix.is_empty:
        suffix_rep = suffix.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        suffix_rep = suffix
    tqdm.write(f"suffix_rep in evaluate_next_token_candidates: {suffix_rep}")
    #tqdm.write(f"suffix_rep text in evaluate_next_token_candidates: {suffix_rep.text}")

    # compute the losses on each sample
    merged = MergedSeq(seqs=[instruct_rep, suffix_rep, q_sample_seq])
    full_instruct = Seq(
        text=merged.to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )
    tqdm.write(f"full_instruct text in evaluate_next_token_candidates: {full_instruct.text}")

    with torch.no_grad():
        target_llm_tf_q = target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=full_instruct,
            target=target_rep,
            loss_params=dict(hard_labels=True, reweight_loss=cfg.reweight_loss),
        )

    loss_batch = target_llm_tf_q.loss_batch.to(next_dist_seq_prompter.device)
    tqdm.write(f"loss_batch in evaluate_next_token_candidates: {loss_batch}")
    tqdm.write(f"loss_batch shape in evaluate_next_token_candidates: {loss_batch.shape}")

    losses = torch.reshape(loss_batch, (ebs, num_samples_per_beam))
    tqdm.write(f"losses in evaluate_next_token_candidates: {losses}")
    tqdm.write(f"losses shape in evaluate_next_token_candidates: {losses.shape}")

    loss_delta = losses - prev_losses[:, None]  # (ebs, num_samples_per_beam)
    tqdm.write(f"loss_delta in evaluate_next_token_candidates: {loss_delta}")
    tqdm.write(f"loss_delta shape in evaluate_next_token_candidates: {loss_delta.shape}")

    next_dist_logprobs_basemodel = next_dist_seq_basemodel.logprobs.squeeze(1)
    tqdm.write(f"next_dist_logprobs_basemodel in evaluate_next_token_candidates: {next_dist_logprobs_basemodel}")
    tqdm.write(f"next_dist_seq_basemodel text in evaluate_next_token_candidates: {next_dist_seq_basemodel.text}")
    tqdm.write(f"next_dist_logprobs_basemodel shape in evaluate_next_token_candidates: {next_dist_logprobs_basemodel.shape}")

    selected_logprobs_basemodel = torch.gather(
        next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids
    )  # (ebs, num_samples_per_beam)
    tqdm.write(f"selected_logprobs_basemodel in evaluate_next_token_candidates: {selected_logprobs_basemodel}")
    tqdm.write(f"selected_logprobs_basemodel shape in evaluate_next_token_candidates: {selected_logprobs_basemodel.shape}")

    factor = cfg.train.q_params.lambda_val
    beam_scores_delta = selected_logprobs_basemodel - loss_delta * factor
    tqdm.write(f"beam_scores_delta in evaluate_next_token_candidates: {beam_scores_delta}")
    tqdm.write(f"beam_scores_delta shape in evaluate_next_token_candidates: {beam_scores_delta.shape}")

    new_beam_scores = prev_beam_scores[:, None] + beam_scores_delta
    tqdm.write(f"new_beam_scores in evaluate_next_token_candidates: {new_beam_scores}")
    tqdm.write(f"new_beam_scores shape in evaluate_next_token_candidates: {new_beam_scores.shape}")

    return new_beam_scores, losses


@torch.no_grad()
def select_next_beams(
    cfg,
    suffix_beams,
    next_token_candidate_ids,
    candidate_beam_scores,
    candidate_losses,
    num_beams_in,
    num_beams_out,
):
    tqdm.write(f"suffix_beams in advPrompterOpt: {suffix_beams}")
    tqdm.write(f"num_beams_in in advPrompterOpt: {num_beams_in}")
    tqdm.write(f"num_beams_out in advPrompterOpt: {num_beams_out}")


    ebs, num_samples_per_beam = candidate_beam_scores.shape
    bs = ebs // num_beams_in
    tqdm.write(f"ebs in advPrompterOpt: {ebs}")
    tqdm.write(f"num_samples_per_beam in advPrompterOpt: {num_samples_per_beam}")
    tqdm.write(f"bs in advPrompterOpt: {bs}")

    candidate_beam_scores = candidate_beam_scores.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    tqdm.write(f"candidate_beam_scores in advPrompterOpt: {candidate_beam_scores}")
    tqdm.write(f"candidate_beam_scores shape in advPrompterOpt: {candidate_beam_scores.shape}")

    if cfg.train.q_params.beams.do_sample:

        if cfg.train.q_params.beams.always_include_best:

            candidate_beam_scores_top_ids = candidate_beam_scores.argmax(dim=-1)
            tqdm.write(f"candidate_beam_scores_top_ids in advPrompterOpt: {candidate_beam_scores_top_ids}")
            tqdm.write(f"candidate_beam_scores_top_ids shape in advPrompterOpt: {candidate_beam_scores_top_ids.shape}")

            candidate_beam_scores_onehot = torch.zeros_like(candidate_beam_scores)
            tqdm.write(f"candidate_beam_scores_onehot in advPrompterOpt: {candidate_beam_scores_onehot}")
            tqdm.write(f"candidate_beam_scores_onehot shape in advPrompterOpt: {candidate_beam_scores_onehot.shape}")

            candidate_beam_scores_onehot.scatter_(
                1, candidate_beam_scores_top_ids[:, None], 1
            )

            candidate_beam_scores_corrected = (
                candidate_beam_scores - 1e10 * candidate_beam_scores_onehot
            )
            tqdm.write(f"candidate_beam_scores_corrected in advPrompterOpt function: {candidate_beam_scores_corrected}")
            tqdm.write(f"candidate_beam_scores_corrected shape in advPrompterOpt function: {candidate_beam_scores_corrected.shape}")

            beam_probs = torch.softmax(
                candidate_beam_scores_corrected / cfg.train.q_params.beams.temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
            tqdm.write(f"beam_probs in advPrompterOpt: {beam_probs}")
            tqdm.write(f"beam_probs shape in advPrompterOpt: {beam_probs.shape}")


        else:
            beam_probs = torch.softmax(
                candidate_beam_scores / cfg.train.q_params.beams.temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
            tqdm.write(f"beam_probs in advPrompterOpt2: {beam_probs}")

        next_beam_indices = beam_probs.multinomial(
            num_samples=num_beams_out, replacement=False
        )  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]
        tqdm.write(f"next_beam_indices in advPrompterOpt: {next_beam_indices}")

        if cfg.train.q_params.beams.always_include_best:
            next_beam_indices = torch.cat(
                [candidate_beam_scores_top_ids[:, None], next_beam_indices[:, :-1]],
                dim=-1,
            )
        tqdm.write(f"next_beam_indices in advPrompterOpt2: {next_beam_indices}")
    
    else:
        next_beam_indices = candidate_beam_scores.topk(
            k=num_beams_out, dim=-1, sorted=True
        ).indices  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]
        tqdm.write(f"Here????????????????")

    next_beam_indices_expanded = (
        next_beam_indices
        + torch.arange(0, bs, device=suffix_beams.device)[:, None]
        * num_beams_in
        * num_samples_per_beam
    )  # (bs, num_beams_out)
    tqdm.write(f"next_beam_indices_expanded in advPrompterOpt: {next_beam_indices_expanded}")
    tqdm.write(f"next_beam_indices_expanded shape in advPrompterOpt: {next_beam_indices_expanded.shape}")

    next_beam_indices_expanded = next_beam_indices_expanded.reshape(-1)
    tqdm.write(f"next_beam_indices_expanded in advPrompterOpt22: {next_beam_indices_expanded}")
    tqdm.write(f"next_beam_indices_expanded shape in advPrompterOpt22: {next_beam_indices_expanded.shape}")

    next_token_candidate_seq = Seq(
        ids=next_token_candidate_ids.reshape(
            bs * num_beams_in * num_samples_per_beam, 1
        ),
        tokenizer=suffix_beams.tokenizer,
        device=suffix_beams.device,
    )
    tqdm.write(f"next_token_candidate_seq logits in advPrompterOpt33: {next_token_candidate_seq.logits}")
    tqdm.write(f"next_token_candidate_seq logits shape in advPrompterOpt33: {next_token_candidate_seq.logits.shape}")
    tqdm.write(f"next_token_candidate_seq text in advPrompterOpt33: {next_token_candidate_seq.text}")


    if suffix_beams.is_empty:
        next_suffix_beams = next_token_candidate_seq[next_beam_indices_expanded]
        tqdm.write(f"next_suffix_beams in advPrompterOpt33: {next_suffix_beams}")

    else:
        beam_candidates = suffix_beams.repeat_interleave(num_samples_per_beam, dim=0)
        tqdm.write(f"beam_candidates logits in advPrompterOpt3355: {beam_candidates.logits}")
        tqdm.write(f"beam_candidates logits shape in advPrompterOpt3355: {beam_candidates.logits.shape}")
        tqdm.write(f"beam_candidates text in advPrompterOpt3355: {beam_candidates.text}")

        beam_candidates.append(next_token_candidate_seq)
        next_suffix_beams = beam_candidates[next_beam_indices_expanded]
        tqdm.write(f"next_suffix_beams logits in advPrompterOpt3344: {next_suffix_beams.logits}")
        tqdm.write(f"next_suffix_beams logits shape in advPrompterOpt3344: {next_suffix_beams.logits.shape}")
        tqdm.write(f"next_suffix_beams text in advPrompterOpt3344: {next_suffix_beams.text}")


    candidate_losses = candidate_losses.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    tqdm.write(f"candidate_losses in advPrompterOpt3344: {candidate_losses}")
    tqdm.write(f"candidate_losses shape in advPrompterOpt3344: {candidate_losses.shape}")

    selected_losses = candidate_losses.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    tqdm.write(f"selected_losses in advPrompterOpt334455: {selected_losses}")
    tqdm.write(f"selected_losses shape in advPrompterOpt334455: {selected_losses.shape}")

    selected_losses = selected_losses.reshape(bs * num_beams_out).detach()
    tqdm.write(f"selected_losses in advPrompterOpt334488: {selected_losses}")
    tqdm.write(f"selected_losses shape in advPrompterOpt334488: {selected_losses.shape}")


    selected_beam_scores = candidate_beam_scores.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    tqdm.write(f"selected_beam_scores in advPrompterOpt33448899: {selected_beam_scores}")
    tqdm.write(f"selected_beam_scores shape in advPrompterOpt33448899: {selected_beam_scores.shape}")

    selected_beam_scores = selected_beam_scores.reshape(bs * num_beams_out).detach()
    tqdm.write(f"selected_beam_scores in advPrompterOpt3344889900: {selected_beam_scores}")
    tqdm.write(f"selected_beam_scores shape in advPrompterOpt3344889900: {selected_beam_scores.shape}")

    return next_suffix_beams, selected_losses, selected_beam_scores


@torch.no_grad()
def evaluate_prompt(
    cfg,
    instruct,
    suffix,
    full_instruct,
    target,
    prompter,
    target_llm,
    generate_target_llm_response,
    print_idx=0,
):
    basemodel_tf = None
    if suffix is not None and not suffix.is_empty:
        basemodel_tf = prompter.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=instruct,
            suffix=suffix,
            use_basemodel=True,
            loss_params=dict(hard_labels=True),
        )

    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(
            hard_labels=True,
            reweight_loss=cfg.reweight_loss,
        ),
    )
    #if cfg.verbose:
    tqdm.write(f" --- Query[{print_idx}]: '{target_llm_tf.query.text[print_idx]}'")
    tqdm.write(f" --- Suffix[{print_idx}]: '{suffix.text[print_idx]}'")
    tqdm.write(f" --- Target[{print_idx}]: '{target.text[print_idx]}'")
    tqdm.write(
            f" --- TF Response[{print_idx}]: '{target_llm_tf.response_dist.text[print_idx]}'"
        )

    if generate_target_llm_response:
        target_llm_ar = target_llm.generate_autoregressive(
            key="target",
            full_instruct=full_instruct,
        )
        #if cfg.verbose:
        tqdm.write(
                f" --- AR Response[{print_idx}]: '{target_llm_ar.response_sample.text[print_idx]}'"
            )
    else:
        target_llm_ar = None

    #if cfg.verbose:
    tqdm.write(f" Evaluating suffix completed. TF Loss: {target_llm_tf.loss:.3f}")

    return target_llm_tf, target_llm_ar, basemodel_tf

@torch.no_grad()
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        tqdm.write(f"sorted_logits in top_p is : {sorted_logits}")
        tqdm.write(f"sorted_logits shape in top_p is : {sorted_logits.shape}")
        tqdm.write(f"sorted_indices in top_p is : {sorted_indices}")
        tqdm.write(f"sorted_indices shape in top_p is : {sorted_indices.shape}")

        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        tqdm.write(f"cumulative_probs in top_p is : {cumulative_probs}")
        tqdm.write(f"cumulative_probs shape in top_p is : {cumulative_probs.shape}")

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        tqdm.write(f"sorted_indices_to_remove in top_p is : {sorted_indices_to_remove}")
        tqdm.write(f"sorted_indices_to_remove shape in top_p is : {sorted_indices_to_remove.shape}")
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        tqdm.write(f"indices_to_remove in top_p is : {indices_to_remove}")
        tqdm.write(f"indices_to_remove shape in top_p is : {indices_to_remove.shape}")
        logits[indices_to_remove] = filter_value
    
        tqdm.write(f"logits shape in top_p is : {logits.shape}")

    return logits
