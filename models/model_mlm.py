

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
	AutoModel,
	BertPreTrainedModel,
	BertModel
)
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import (
	SequenceClassifierOutput,
	MaskedLMOutput,
	ModelOutput
)
from transformers.file_utils import (
	add_code_sample_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	replace_return_docstrings,
)


@dataclass
class MultiTaskOutput(ModelOutput):
	"""
	Class for multi-task training outputs.

	Args:
		total_loss (torch.FloatTensor):
			Summed masked language modeling (MLM) loss.

		base_loss (torch.FloatTensor):
			Base masked language modeling (MLM) loss coming from predicting the speaker of the target utterance.

		aux_loss (torch.FloatTensor):
			Auxiliary masked language modeling (MLM) loss coming from predicting the speakers of the neighbouring utterances.

		base_scores (List(torch.FloatTensor)):
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) on the candidate
			speakers of the target utterance.
		
		auxiliary_scores (List(torch.FloatTensor)):
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) on the candidate
			speakers of the neighbouring utterances.

		hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
			one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

		attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.
	"""

	total_loss: torch.FloatTensor = None
	base_loss: torch.FloatTensor = None
	neighbour_speaker_loss: torch.FloatTensor = None
	role_prediction_loss: torch.FloatTensor = None
	base_scores: List[torch.FloatTensor] = None
	neighbour_speaker_prediction_scores: List[torch.FloatTensor] = None
	role_prediction_scores: List[torch.FloatTensor] = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""
	This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
	library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
	etc.)
	This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
	Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
	and behavior.
	Parameters:
		config ([`BertConfig`]): Model configuration class with all the parameters of the model.
			Initializing with a config file does not load the weights associated with the model, only the
			configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
	Args:
		input_ids (`torch.LongTensor` of shape `({0})`):
			Indices of input sequence tokens in the vocabulary.
			Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
			[`PreTrainedTokenizer.__call__`] for details.
			[What are input IDs?](../glossary#input-ids)
		attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
			Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
			- 1 for tokens that are **not masked**,
			- 0 for tokens that are **masked**.
			[What are attention masks?](../glossary#attention-mask)
		token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
			Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
			1]`:
			- 0 corresponds to a *sentence A* token,
			- 1 corresponds to a *sentence B* token.
			[What are token type IDs?](../glossary#token-type-ids)
		position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
			Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
			config.max_position_embeddings - 1]`.
			[What are position IDs?](../glossary#position-ids)
		head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
			Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
			- 1 indicates the head is **not masked**,
			- 0 indicates the head is **masked**.
		inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
			Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
			is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
			model's internal embedding lookup matrix.
		output_attentions (`bool`, *optional*):
			Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
			tensors for more detail.
		output_hidden_states (`bool`, *optional*):
			Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
			more detail.
		return_dict (`bool`, *optional*):
			Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings("""Bert Model with a `language modeling` head on top for speaker identification in novel dialogues. """, BERT_START_DOCSTRING)
class SindBertMaskedLM(BertPreTrainedModel):

	_keys_to_ignore_on_load_unexpected = [r"pooler"]
	_keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config, add_pooling_layer=False)
		self.cls = BertOnlyMLMHead(config)

		self.init_weights()

	def get_output_embeddings(self):
		return self.cls.predictions.decoder

	def set_output_embeddings(self, new_embeddings):
		self.cls.predictions.decoder = new_embeddings

	# @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
	# @add_code_sample_docstrings(
	#     processor_class=_TOKENIZER_FOR_DOC,
	#     checkpoint=_CHECKPOINT_FOR_DOC,
	#     output_type=MaskedLMOutput,
	#     config_class=_CONFIG_FOR_DOC,
	# )
	def forward(
		self,
		bert_inputs,
		target_token_ids,
		mask_pos,
		label_idx,
		neighbour_utter_mask_poses,
		neighbour_utter_label_idxes,
		narr_role_mask_poses,
		narr_role_label_idxes,
		args
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
			config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
			(masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
		"""

		input_ids = bert_inputs['input_ids']
		attention_mask = bert_inputs['attention_mask']
		token_type_ids = bert_inputs['token_type_ids']

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=None,
			head_mask=None,
			inputs_embeds=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None
		)

		sequence_output = outputs[0]
		prediction_scores = self.cls(sequence_output)

		# select the masked position
		batch_base_score_pairs = []
		batch_neighbour_score_pairs = []
		batch_role_score_pairs = []
		batch_base_scores = []
		batch_neighbour_speaker_scores = []
		batch_narr_role_scores = []
		for i in range(len(label_idx)):
			sent_scores_i = prediction_scores[i]
			target_token_ids_i = target_token_ids[i]
			mask_pos_i = mask_pos[i]
			neighbour_speaker_mask_poses_i = neighbour_utter_mask_poses[i]
			role_mask_poses_i = narr_role_mask_poses[i]
			label_idx_i = label_idx[i]
			neighbour_speaker_label_idxes_i = neighbour_utter_label_idxes[i]
			role_label_idxes_i = narr_role_label_idxes[i]

			def from_hidden_to_scores(sent_scores, mask_idx, target_token_idxes, label_idx):
				target_scores = torch.index_select(sent_scores[mask_idx, :], dim=0, index=target_token_idxes)
				speaker_score = target_scores[label_idx].expand(target_token_idxes.size(0) - 1)
				candidate_scores = torch.cat([target_scores[:label_idx], target_scores[label_idx + 1:]], dim=0)
				score_pairs = torch.stack([speaker_score, candidate_scores], dim=1)
				return score_pairs, target_scores

			target_token_ids_i = torch.tensor(target_token_ids_i).to(input_ids.device)
			inst_base_score_pairs, inst_base_scores = from_hidden_to_scores(sent_scores_i, mask_pos_i, target_token_ids_i, label_idx_i)
			batch_base_score_pairs.append(inst_base_score_pairs)
			batch_base_scores.append(inst_base_scores)
			
			# prepare neighbour speaker score pairs for computing neighbour speaker prediction loss
			for aux_mask_pos, aux_label_idx in zip(neighbour_speaker_mask_poses_i, neighbour_speaker_label_idxes_i):
				aux_score_pairs_i, inst_aux_scores = from_hidden_to_scores(sent_scores_i, aux_mask_pos, target_token_ids_i, aux_label_idx)
				batch_neighbour_score_pairs.append(aux_score_pairs_i)
				batch_neighbour_speaker_scores.append(inst_aux_scores)

			# prepare role prediction score pairs for computing role prediction loss
			for aux_mask_pos, aux_label_idx in zip(role_mask_poses_i, role_label_idxes_i):
				aux_score_pairs_i, inst_aux_scores = from_hidden_to_scores(sent_scores_i, aux_mask_pos, target_token_ids_i, aux_label_idx)
				batch_role_score_pairs.append(aux_score_pairs_i)
				batch_narr_role_scores.append(inst_aux_scores)
		
		batch_base_score_pairs = torch.cat(batch_base_score_pairs, dim=0)
		
		if batch_base_score_pairs.size(0) > 0:
			loss_fct = nn.MarginRankingLoss(margin=args.margin)
			base_loss = loss_fct(batch_base_score_pairs[:, 0], batch_base_score_pairs[:, 1], torch.ones_like(batch_base_score_pairs[:, 0]))
		else:
			base_loss = torch.tensor(0.0, dtype=torch.float32, device=input_ids.device)

		if batch_neighbour_score_pairs:
			batch_neighbour_score_pairs = torch.cat(batch_neighbour_score_pairs, dim=0)
			neighbour_speaker_loss = loss_fct(batch_neighbour_score_pairs[:, 0], batch_neighbour_score_pairs[:, 1], torch.ones_like(batch_neighbour_score_pairs[:, 0]))
		else:
			neighbour_speaker_loss = torch.tensor(0.0, dtype=torch.float32, device=base_loss.device)

		if batch_role_score_pairs:
			batch_role_score_pairs = torch.cat(batch_role_score_pairs, dim=0)
			role_prediction_loss = loss_fct(batch_role_score_pairs[:, 0], batch_role_score_pairs[:, 1], torch.ones_like(batch_role_score_pairs[:, 0]))
		else:
			role_prediction_loss = torch.tensor(0.0, dtype=torch.float32, device=base_loss.device)

		total_loss = base_loss
		if args.lbd1 > 0.0:
			total_loss += args.lbd1 * neighbour_speaker_loss
		elif args.lbd2 > 0.0:
			total_loss += + args.lbd2 * role_prediction_loss

		return MultiTaskOutput(
			total_loss=total_loss,
			base_loss=base_loss,
			neighbour_speaker_loss=neighbour_speaker_loss,
			role_prediction_loss=role_prediction_loss,
			base_scores=batch_base_scores,
			neighbour_speaker_prediction_scores=batch_neighbour_speaker_scores,
			role_prediction_scores=batch_narr_role_scores,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

