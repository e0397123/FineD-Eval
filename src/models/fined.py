from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils import data_utils, logging
from src.models.trainer import Trainer

logger = logging.get_logger(__name__)

@dataclass
class DialogueOutputs:
    loss: torch.Tensor = field(default=None)
    log_probs_1: torch.Tensor = field(default=None)
    log_probs_2: torch.Tensor = field(default=None)
    label_tensor: torch.Tensor = field(default=None)
    details: Any = field(default=None)
    name: Any = field(default=0)

class DialogueModel(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.head = nn.Linear(self.model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.hinge_loss = torch.nn.MarginRankingLoss(reduction='none', margin=0.1)

    def forward(self, **inputs):
        labels = inputs.pop("labels") if "labels" in inputs else None
        # dial_lengths = inputs.pop("dial_len")
        model_inputs_1 = dict()
        model_inputs_1['input_ids'] = inputs['input_ids_1']
        model_inputs_1['attention_mask'] = inputs['input_masks_1']
        model_inputs_2 = dict()
        model_inputs_2['input_ids'] = inputs['input_ids_2']
        model_inputs_2['attention_mask'] = inputs['input_masks_2']
        if not "Roberta" in self.model.__class__.__name__:
            model_inputs_1['token_type_ids'] = inputs['token_type_ids_1']
            model_inputs_2['token_type_ids'] = inputs['token_type_ids_2']
        model_outputs_1 = self.model(**model_inputs_1, return_dict=True, output_hidden_states=True)
        last_hidden_state_1 = model_outputs_1.hidden_states[-2]
        last_hidden_state_1 = self.dropout(last_hidden_state_1)

        model_outputs_2 = self.model(**model_inputs_2, return_dict=True, output_hidden_states=True)
        last_hidden_state_2 = model_outputs_2.hidden_states[-2]
        last_hidden_state_2 = self.dropout(last_hidden_state_2)
        
        pooled_rep_1 = torch.mean(last_hidden_state_1, dim=1)
        pooled_rep_2 = torch.mean(last_hidden_state_2, dim=1)
        logits_1 = torch.squeeze(self.sigmoid(self.head(pooled_rep_1)), dim=-1)
        logits_2 = torch.squeeze(self.sigmoid(self.head(pooled_rep_2)), dim=-1)

        loss_coh_ixs = torch.add(torch.add(labels * (-1), torch.ones(labels.size()).to(self.model.device)) * 2,
                                 torch.ones(labels.size()).to(self.model.device) * (-1))
        coh_loss = self.hinge_loss(logits_1, logits_2, loss_coh_ixs)

        outputs = DialogueOutputs(log_probs_1=logits_1, log_probs_2=logits_2, label_tensor=labels)
        if labels is not None:
            outputs.loss = coh_loss
        return outputs
    

class MultiHeadModelDialogue(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.names = args.train_on
        self.parallel = args.parallel
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(self.model.config.hidden_size, 1)
                for name in self.names
            }
        )
        logger.info(f"adding {len(self.names)} heads")
        self.hinge_loss = torch.nn.MarginRankingLoss(reduction='none', margin=0.1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, **inputs):
        if self.parallel:
            return self.parallel_forward(**inputs)
        return self.batch_forward(**inputs)

    def batch_forward(self, **inputs):
        labels = inputs.pop("labels") if "labels" in inputs else None
        dial_lengths = inputs.pop("dial_len")
        datasets = inputs.pop("datasets")
        model_inputs_1 = dict()
        model_inputs_1['input_ids'] = inputs['input_ids_1']
        model_inputs_1['attention_mask'] = inputs['input_masks_1']
        model_inputs_2 = dict()
        model_inputs_2['input_ids'] = inputs['input_ids_2']
        model_inputs_2['attention_mask'] = inputs['input_masks_2']
        if not "Roberta" in self.model.__class__.__name__:
            model_inputs_1['token_type_ids'] = inputs['token_type_ids_1']
            model_inputs_2['token_type_ids'] = inputs['token_type_ids_2']
        batch_sizes = list(Counter(datasets).values())
        adapters = list(Counter(datasets).keys())

        model_outputs_1 = self.model(**model_inputs_1, return_dict=True, output_hidden_states=True)
        last_hidden_state_1 = model_outputs_1.hidden_states[-2]
        last_hidden_state_1 = self.dropout(last_hidden_state_1)
        pooled_rep_1 = torch.mean(last_hidden_state_1, dim=1)

        model_outputs_2 = self.model(**model_inputs_2, return_dict=True, output_hidden_states=True)
        last_hidden_state_2 = model_outputs_2.hidden_states[-2]
        last_hidden_state_2 = self.dropout(last_hidden_state_2)
        pooled_rep_2 = torch.mean(last_hidden_state_2, dim=1)

        parts_1 = pooled_rep_1.split(batch_sizes, dim=0)
        dial_lengths_parts = dial_lengths.split(batch_sizes, dim=0)
        logit_lst_1 = []
        for name, hidden_state, d_len in zip(adapters, parts_1, dial_lengths_parts):
            # logit_lst.append(self.gru[name+'_gru'](d_len, self.heads[name](hidden_state)))
            logit_lst_1.append(self.heads[name](hidden_state))
        logits_1 = torch.cat(logit_lst_1, dim=0)
        logits_1 = torch.squeeze(self.sigmoid(logits_1), dim=-1)

        parts_2 = pooled_rep_2.split(batch_sizes, dim=0)
        logit_lst_2 = []
        for name, hidden_state, d_len in zip(adapters, parts_2, dial_lengths_parts):
            # logit_lst.append(self.gru[name+'_gru'](d_len, self.heads[name](hidden_state)))
            logit_lst_2.append(self.heads[name](hidden_state))
        logits_2 = torch.cat(logit_lst_2, dim=0)
        logits_2 = torch.squeeze(self.sigmoid(logits_2), dim=-1)

        loss_coh_ixs = torch.add(torch.add(labels * (-1), torch.ones(labels.size()).to(self.model.device)) * 2,
                                 torch.ones(labels.size()).to(self.model.device) * (-1))
        coh_loss = self.hinge_loss(logits_1, logits_2, loss_coh_ixs)

        outputs = DialogueOutputs(log_probs_1=logits_1, log_probs_2=logits_2, label_tensor=labels)
        if labels is not None:
            outputs.loss = coh_loss
        return outputs

    def parallel_forward(self, **inputs):
        labels = inputs.pop("labels") if "labels" in inputs else None
        dial_lengths = inputs.pop("dial_len")
        model_inputs_1 = dict()
        model_inputs_1['input_ids'] = inputs['input_ids_1']
        model_inputs_1['attention_mask'] = inputs['input_masks_1']
        model_inputs_2 = dict()
        model_inputs_2['input_ids'] = inputs['input_ids_2']
        model_inputs_2['attention_mask'] = inputs['input_masks_2']
        if not "Roberta" in self.model.__class__.__name__:
            model_inputs_1['token_type_ids'] = inputs['token_type_ids_1']
            model_inputs_2['token_type_ids'] = inputs['token_type_ids_2']
        adapters = self.names
        model_outputs_1 = self.model(**model_inputs_1, return_dict=True, output_hidden_states=True)
        model_outputs_2 = self.model(**model_inputs_2, return_dict=True, output_hidden_states=True)

        B, N = inputs["input_ids_1"].shape
        K = len(adapters)
        hidden_states_1 = model_outputs_1.hidden_states[-2]
        hidden_states_1 = self.dropout(hidden_states_1)
        logit_lst_1 = []
        # (B * K, 1) -> [(B, 1)] x K
        # for adapter, hidden_state in zip(
        #         adapters, torch.mean(hidden_states_1, dim=1).split(B, dim=0)
        # ):
        #     logit_lst_1.append(self.heads[adapter](hidden_state))
        for adapter in adapters:
            logit_lst_1.append(self.heads[adapter](torch.mean(hidden_states_1, dim=1)))

        # (B, K, 1)
        logits_1 = torch.stack(logit_lst_1, dim=1)
        logits_1 = torch.squeeze(self.sigmoid(logits_1), dim=-1)


        B, N = inputs["input_ids_2"].shape
        K = len(adapters)
        hidden_states_2 = model_outputs_2.hidden_states[-2]
        hidden_states_2 = self.dropout(hidden_states_2)
        logit_lst_2 = []
        # (B * K, 1) -> [(B, 1)] x K
        # for adapter, hidden_state in zip(
        #         adapters, torch.mean(hidden_states_2, dim=1).split(B, dim=0)
        # ):
        #     logit_lst_2.append(self.heads[adapter](hidden_state))
        for adapter in adapters:
            logit_lst_2.append(self.heads[adapter](torch.mean(hidden_states_2, dim=1)))
        # (B, K, 1)
        logits_2 = torch.stack(logit_lst_2, dim=1)
        logits_2 = torch.squeeze(self.sigmoid(logits_2), dim=-1)

        expert_outputs = [
            DialogueOutputs(
                log_probs_1=torch.squeeze(s_1),
                log_probs_2=torch.squeeze(s_2),
                name=a,
            )
            for s_1, s_2, a in zip(
                logits_1.split(1, dim=1),
                logits_2.split(1, dim=1),
                adapters,
            )
        ]
        # (B)
        gated_log_probs_1 = torch.mean(logits_1, dim=1)
        gated_log_probs_2 = torch.mean(logits_2, dim=1)

        expert_outputs.append(
            DialogueOutputs(
                log_probs_1=gated_log_probs_1,
                log_probs_2=gated_log_probs_2,
                label_tensor=labels
            )
        )

        loss_coh_ixs = torch.add(torch.add(labels * (-1), torch.ones(labels.size()).to(self.model.device)) * 2,
                                 torch.ones(labels.size()).to(self.model.device) * (-1))

        if labels is not None:
            lst = (expert_outputs if "is_prediction" in inputs else [expert_outputs[-1]])
            for o in lst:
                o.loss = self.hinge_loss(o.log_probs_1, o.log_probs_2, loss_coh_ixs)

        outputs = expert_outputs.pop()
        outputs.details = {
            "expert_outputs": expert_outputs,
        }

        return outputs


class FineTrainer(Trainer):
    def initialize(self, args):
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": data_utils.dialog_tokens()}
        )
        if args.multi_head:
            model = MultiHeadModelDialogue(tokenizer, args)
            logger.info('loading the multi-task model')
        else:
            model = DialogueModel(tokenizer, args)
            logger.info('loading single model')
        return tokenizer, model

    def get_predictions(self, args, model, batch, outputs, tokenizer):
        predictions = []
        if outputs.details:
            expert_outputs = outputs.details["expert_outputs"]
            for i, e in enumerate(batch["examples"]):
                if outputs.log_probs_1[i].item() < outputs.log_probs_2[i].item():
                    s = 1
                else:
                    s = 0
                p = {
                    "e": e,
                    "label": outputs.label_tensor[i].item(),
                    "score": s,
                    "loss": outputs.loss[i].item(),
                    "pred": (outputs.log_probs_1[i].item(),outputs.log_probs_2[i].item()),
                    "experts_pred": ([x.log_probs_1[i].item() for x in expert_outputs],
                                     [x.log_probs_2[i].item() for x in expert_outputs]),
                    "experts_name": [x.name for x in expert_outputs],
                    "experts_loss": [x.loss[i].item() for x in expert_outputs],
                    "name": outputs.name,
                }
                predictions.append(p)

        else:
            for i, e in enumerate(batch["examples"]):
                if outputs.log_probs_1[i].item() < outputs.log_probs_2[i].item():
                    s = 1
                else:
                    s = 0
                p = {
                    "e": e,
                    "label": outputs.label_tensor[i].item(),
                    "score": s,
                    "loss": outputs.loss[i].item(),
                    "pred": (outputs.log_probs_1[i].item(),outputs.log_probs_2[i].item()),
                    "name": outputs.name,
                }
                predictions.append(p)
        return predictions

    def get_inputs(self, args, batch, **kwargs):
        inputs = {
            k: batch[k]
            for k in ("input_ids_1", "input_masks_1", "token_type_ids_1",
                      "input_ids_2", "input_masks_2", "token_type_ids_2",
                      "labels", "dial_len")
        }
        if args.multi_head:
            inputs["datasets"] = [e["dataset"] for e in batch["examples"]]
        return inputs
