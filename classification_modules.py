# (c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import GPT2Model, GPT2PreTrainedModel, GPT2LMHeadModel
from transformers import BartModel, BartPretrainedModel, BartForConditionalGeneration
from torch.nn import Sigmoid, Softmax


logger = logging.getLogger(__name__)


class ConcatSummary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim * 7, 1)  # hiddensize, numclasses

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class Summary(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(emb_dim, 1)  # hiddensize, numclasses

    def forward(self, output):
        dropout_pooled_output = self.dropout(output)
        logits = self.summary(dropout_pooled_output)
        return logits


class BARTPK_ctxt(BartForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        # config.vocab_size = config.vocab_size + 4
        self.model = BartModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.concat_summary = ConcatSummary(emb_dim=config.d_model)
        self.summary = Summary(emb_dim=config.d_model)
        self.attn1 = nn.Linear(config.d_model, 5)
        self.attn2 = nn.Linear(5, config.d_model)  # Selected knowledge 개수만
        self.max_position = config.max_position_embeddings
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        input_eos=None,
        only_dial_input_ids=None,
        decoder_input_ids=None,
        persona_input_ids=None,
        knowledge_input_ids=None,
        persona_can_idx=None,
        persona_grounding=None,
        knowledge_can_idx=None,
        knowledge_grounding=None,
        tot_knowledge=None,
        tot_knowledge_eos=None,
        training=None,
        lm_labels=None,
        mc_token_ids=None,
    ):

        # machine = 50265
        # human = 50266
        persona = 50267
        knowledge = 50268
        padding = 1
        bos = 0
        eos = 2
        num_chosen_paragraph = 5
        device = input_ids.get_device()

        persona_tensor = torch.tensor([persona]).cuda(device)
        knowledge_tensor = torch.tensor([knowledge]).cuda(device)
        bos_tensor = torch.tensor([bos]).cuda(device)
        eos_tensor = torch.tensor([eos]).cuda(device)

        outputs = tuple()
        dynamic_lm_logits = None
        persona_logits = None
        knowledge_logits = None

        if input_eos is not None:
            lm_hidden_states = self.model(input_ids=input_ids)["last_hidden_state"]
            batch, seq_len, embdim = lm_hidden_states.size()
            lm_hidden_states_eos_list = []
            # получаем последние токены входящей последовательности
            # такой подход используется в барт для классификации
            # последовательностей
            for i in range(batch):
                lm_hidden_states_batch = lm_hidden_states[i]
                lm_eos_batch = input_eos[i]
                lm_hidden_states_eos = torch.index_select(
                    lm_hidden_states_batch, -2, lm_eos_batch
                )
                lm_hidden_states_eos_list.append(lm_hidden_states_eos)
            lm_eos_rep = torch.stack(lm_hidden_states_eos_list)
            # получаем репрезентацию tot_knowledge
            tot_knowledge_hidden_states = self.model(
                input_ids=tot_knowledge.view(batch * num_chosen_paragraph, -1)
            )["last_hidden_state"].view(batch, num_chosen_paragraph, -1, embdim)
            tot_knowledge_eos_list = []
            for i in range(batch):
                tot_knowledge_hidden_states_batch = tot_knowledge_hidden_states[i]
                tot_knowledge_eos_batch = tot_knowledge_eos[i]
                tot_knowledge_eos_list_batch = []
                for j in range(5):
                    tot_knowledge_eos_token = torch.index_select(
                        tot_knowledge_hidden_states_batch[j],
                        -2,
                        tot_knowledge_eos_batch[j],
                    )
                    tot_knowledge_eos_list_batch.append(
                        tot_knowledge_eos_token.squeeze()
                    )
                tot_knowledge_eos_batch_rep = torch.stack(tot_knowledge_eos_list_batch)
                tot_knowledge_eos_list.append(tot_knowledge_eos_batch_rep)

            tot_knowledge_eos_final = torch.stack(tot_knowledge_eos_list)
            knowledge_inctxt_attn = self.attn1(tot_knowledge_eos_final)
            knowledge_inctxt_eos_rep = self.attn2(knowledge_inctxt_attn)

            # конкатенируем тотальные знания и репрезентацию последних токенов
            inctxt_states = torch.cat(
                (lm_eos_rep, knowledge_inctxt_eos_rep), dim=1
            ).type_as(input_ids)

            # persona candidates
            num_persona_can = 5
            if persona_input_ids is not None:
                if persona_can_idx is not None:
                    pass
                    # получаем предсказание персоны
                    persona_logits = torch.randn((batch, num_persona_can)).to(
                        device=device
                    )
                    outputs = (persona_logits,)

            # knowledge candidates
            num_knowledge_can = 10
            if knowledge_input_ids is not None:
                # получаем репрезентацию знаний
                knowledge_emb = self.model(
                    input_ids=knowledge_input_ids.view(batch * num_knowledge_can, -1)
                )["last_hidden_state"].view(batch, num_knowledge_can, -1, embdim)
                if knowledge_can_idx is not None:
                    knowledge_list = []
                    for batch_i in range(batch):
                        inctxt_eos_batch = inctxt_states[batch_i]
                        knowledge_emb_batch = knowledge_emb[batch_i]
                        knowledge_can_idx_batch = knowledge_can_idx[batch_i]
                        knowledge_batch_list = []
                        for i in range(num_knowledge_can):
                            # получаем репрезентацию знаний о кандидате
                            knowledge_selected = torch.index_select(
                                knowledge_emb_batch[i], 0, knowledge_can_idx_batch[i]
                            )
                            # итоговая репрезентация знаний о кандидате
                            final_rep_knowledge = torch.cat(
                                [
                                    inctxt_eos_batch.type_as(lm_eos_rep),
                                    knowledge_selected.type_as(lm_eos_rep),
                                ],
                                dim=0,
                            )
                            knowledge_batch_list.append(final_rep_knowledge)
                        knowledge_batch_list = torch.stack(knowledge_batch_list)
                        knowledge_list.append(knowledge_batch_list)
                    # итоговая репрезентация знаний всех кандидатов
                    knowledge_rep = torch.stack(knowledge_list).view(
                        batch * num_knowledge_can, -1
                    )
                    # получаем предсказание знаний о кандидате
                    knowledge_logits = self.concat_summary(knowledge_rep).view(
                        batch, -1
                    )
                    outputs = (knowledge_logits,) + outputs

        dynamic_lm_hidden_states = torch.randn(
            (
                decoder_input_ids.shape[0],
                decoder_input_ids.shape[1],
                self.config.d_model,
            )
        ).to(device)

        if dynamic_lm_hidden_states is not None:
            dynamic_lm_logits = self.lm_head(dynamic_lm_hidden_states)
            outputs = (dynamic_lm_logits,) + outputs

        if persona_grounding is not None:
            loss_fct = BCEWithLogitsLoss()
            persona_loss = loss_fct(
                persona_logits.view(batch, -1),
                persona_grounding.type_as(persona_logits),
            )
            outputs = (persona_loss,) + outputs

        if knowledge_grounding is not None:
            loss_fct = CrossEntropyLoss()
            knowledge_loss = loss_fct(
                knowledge_logits.view(batch, -1), knowledge_grounding
            )
            outputs = (knowledge_loss,) + outputs

        if training is True:
            shift_logits = dynamic_lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs = (lm_loss,) + outputs

        return outputs  # (lm_loss-training), (knowledge_loss), (persona_loss), dynamic_lm_logits, knowledge_logits, persona_logits, persona_detect_logits, presents, (all hidden_states), (attentions)
