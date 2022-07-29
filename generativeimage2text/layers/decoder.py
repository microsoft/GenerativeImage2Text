import warnings
from torch.nn import functional as F
import torch
import logging
from torch import nn
from pprint import pformat
import functools


class TextualHead(nn.Module):
    def __init__(self,
                 visual_feature_size: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    @property
    def textual_feature_size(self):
        return self.hidden_size

def create_projecton_layer(visual_projection_type,
                           visual_feature_size,
                           textual_feature_size,
                           ):
    if visual_projection_type is None:
        visual_projection = nn.Linear(
            visual_feature_size, textual_feature_size
        )
    elif visual_projection_type == 'linearLn':
        visual_projection = nn.Sequential(
            nn.Linear(
                visual_feature_size, textual_feature_size
            ),
            nn.LayerNorm(textual_feature_size),
        )
    else:
        raise NotImplementedError(visual_projection_type)
    return visual_projection

class WordAndPositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        #self.padding_idx = padding_idx

        #self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.words = nn.Embedding(vocab_size, hidden_size)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens):
        position_indices = self._create_position_indices(tokens)

        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        #token_mask = (tokens != self.padding_idx).unsqueeze(-1)
        #embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions

class BertEncoderAsDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, tgt, memory,
                tgt_mask=None,
                #memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_bi_valid_mask=None,
                encoder_history_states=None,
                # tgt_bi_valid_mask: N x num_tgt
                ):
        assert tgt_key_padding_mask is None, 'not supported'
        assert tgt_mask.dim() == 2
        assert tgt_mask.shape[0] == tgt_mask.shape[1]
        # tgt_mask should always be 0/negative infinity
        # mask
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)

        hidden_states = torch.cat((memory, tgt), dim=1)
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        top_right = torch.full((num_memory, num_tgt), float('-inf'), device=tgt.device, dtype=dtype,)
        bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device,)
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # if it is False, it means valid. That is, it is not a padding
        assert memory_key_padding_mask.dtype == torch.bool
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        if tgt_bi_valid_mask is not None:
            # verify the correctness
            bs = full_attention_mask.shape[0]
            # during inference, tgt_bi_valid_mask's length is not changed, but
            # num_tgt can be increased
            max_valid_target = tgt_bi_valid_mask.shape[1]
            mask = tgt_bi_valid_mask[:, None, :].expand((bs, num_memory+num_tgt, max_valid_target))
            full_attention_mask[:, :, num_memory:(num_memory+max_valid_target)][mask] = 0

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]

        if encoder_history_states is None:
            result = self.encoder(
                hidden_states=hidden_states,
                attention_mask=full_attention_mask,
                encoder_history_states=encoder_history_states,
            )
            result = list(result)
            result[0] = result[0][:, num_memory:].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result[0], result[1]
            else:
                # make it back-compatible
                return result[0]
        else:
            encoder_out = self.encoder(
                hidden_states=hidden_states[:, -1:],
                attention_mask=full_attention_mask[:, :, -1:],
                encoder_history_states=encoder_history_states,
            )
            result = encoder_out[0].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result, encoder_out[1]
            else:
                return result

def create_decoder(decoder_type, norm_type,
                   textual_feature_size,
                   attention_heads,
                   feedforward_size,
                   dropout,
                   num_layers,
                   output_hidden_states=False,
                   use_mlp_wrapper=None,
                   ):
    assert norm_type in ['post', 'pre']
    if decoder_type is None:
        assert NotImplemented
    elif decoder_type == 'bert_en':
        from .bert import BertConfig
        from .bert.modeling_bert import BertEncoder
        config = BertConfig(
            vocab_size_or_config_json_file=30522,
            hidden_size=textual_feature_size,
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            intermediate_size=feedforward_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        config.pre_norm=(norm_type == 'pre')
        config.use_mlp_wrapper = use_mlp_wrapper
        config.output_hidden_states = output_hidden_states
        encoder = BertEncoder(config)
        return BertEncoderAsDecoder(encoder)

class AutoRegressiveBeamSearch(object):
    def __init__(
        self,
        eos_index: int,
        max_steps: int = 50,
        beam_size: int = 5,
        per_node_beam_size: int = 2,
        fix_missing_prefix=False,
    ) -> None:
        self._eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.fix_missing_prefix = fix_missing_prefix
        assert fix_missing_prefix, 'should always true'

    def search(self, start_predictions, step,
               only_return_best=True,
               do_sample=False,
               top_k=0,
               top_p=None,
               num_return_sequences=1,
               temperature=1,
               ):
        if num_return_sequences > 1:
            start_predictions = start_predictions[:, None, :].expand(
                start_predictions.shape[0],
                num_return_sequences,
                start_predictions.shape[1])
            start_predictions = start_predictions.reshape(-1, start_predictions.shape[-1])

        batch_size = start_predictions.size()[0]
        if not self.fix_missing_prefix:
            # List of `(batch_size, beam_size, length)` tensors.
            # Does not include the start symbols, which are implicit.
            predictions: torch.Tensor = torch.empty(
                (batch_size, self.beam_size, 0),
                dtype=torch.long, device=start_predictions.device
            )
        else:
            #predictions = start_predictions.unsqueeze(-1).expand((batch_size, self.beam_size, start_predictions.shape[-1]))
            predictions = start_predictions.unsqueeze(1).expand((batch_size, self.beam_size, start_predictions.shape[-1]))
        # Calculate the first timestep. This is done outside the main loop
        # because we are going from a single decoder input (the output from the
        # encoder) to the top `beam_size` decoder outputs. On the other hand,
        # within the main loop we are going from the `beam_size` elements of the
        # beam to `beam_size`^2 candidates from which we will select the top
        # `beam_size` elements for the next iteration.
        # shape: (batch_size, num_classes)
        start_class_logits = step(start_predictions)

        if temperature != 1:
            assert do_sample
            start_class_logits = start_class_logits / temperature

        # Convert logits to logprobs.
        # shape: (batch_size * beam_size, vocab_size)
        start_class_logprobs = F.log_softmax(start_class_logits, dim=1)

        num_classes = start_class_logprobs.size()[1]

        if not do_sample:
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            start_top_logprobs, start_predicted_classes = start_class_logprobs.topk(
                self.beam_size
            )
        else:
            start_predicted_classes = torch.multinomial(start_class_logits.softmax(dim=1),
                    num_samples=self.beam_size)  # (batch_size, num_beams)
            start_top_logprobs = torch.gather(start_class_logprobs, -1, start_predicted_classes)  # (batch_size, num_beams)

        if (
            self.beam_size == 1
            and (start_predicted_classes == self._eos_index).all()
        ):
            warnings.warn(
                "Empty captions predicted. You may want to increase beam "
                "size or ensure your step function is working properly.",
                RuntimeWarning,
            )
            if only_return_best:
                return start_predicted_classes, start_top_logprobs
            else:
                return start_predicted_classes.unsqueeze(-1), start_top_logprobs

        # The log probs for the last time step.
        # shape: (batch_size, beam_size)
        last_logprobs = start_top_logprobs

        # shape: (batch_size, beam_size, sequence_length)
        predictions = torch.cat([predictions, start_predicted_classes.unsqueeze(-1)], dim=-1)

        # Log probability tensor that mandates that the end token is selected.
        # shape: (batch_size * beam_size, num_classes)
        logprobs_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logprobs_after_end[:, self._eos_index] = 0.0

        logits_after_end = start_class_logprobs.new_full(
            (batch_size * self.beam_size, num_classes), float("-inf")
        )
        logits_after_end[:, self._eos_index] = 0

        #for timestep in range(self.max_steps - 1):
        while predictions.shape[-1] < self.max_steps:
            # shape: (batch_size * beam_size,)
            last_predictions = predictions[:, :, -1].reshape(batch_size * self.beam_size)

            # If every predicted token from the last step is `self._eos_index`,
            # then we can stop early.
            if (last_predictions == self._eos_index).all():
                break

            predictions_so_far = predictions.view(
                batch_size * self.beam_size, -1
            )
            # shape: (batch_size * beam_size, num_classes)
            class_logits = step(predictions_so_far)

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            class_logits = class_logits.scatter(1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            # shape: (batch_size * beam_size, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size, num_classes
            )

            # Here we are finding any beams where we predicted the end token in
            # the previous timestep and replacing the distribution with a
            # one-hot distribution, forcing the beam to predict the end token
            # this timestep as well.
            # shape: (batch_size * beam_size, num_classes)
            #cleaned_logprobs = torch.where(
                #last_predictions_expanded == self._eos_index,
                #logprobs_after_end,
                #class_logprobs,
            #)
            class_logits = torch.where(
                last_predictions_expanded == self._eos_index,
                logits_after_end,
                class_logits,
            )

            # Convert logits to logprobs.
            # shape: (batch_size * beam_size, vocab_size)
            #for index in range(batch_size * self.beam_size):
                ##class_logprobs[index, predictions_so_far[index, -1]] = -10000
                #class_logprobs[index, predictions_so_far[index, -1]] = -10000
            class_logprobs = F.log_softmax(class_logits, dim=1)

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            #class_logprobs = class_logprobs.scatter(1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            if not do_sample:
                # shape (both): (batch_size * beam_size, per_node_beam_size)
                top_logprobs, predicted_classes = class_logprobs.topk(
                    self.per_node_beam_size
                )
            else:
                if temperature != 1:
                    class_logits = class_logits / temperature
                #class_logits = top_k_top_p_filtering(class_logits, top_k=top_k, top_p=top_p)
                predicted_classes = torch.multinomial(class_logits.softmax(dim=1),
                        num_samples=self.per_node_beam_size)  # (batch_size * num_beams, TOPN_PER_BEAM)
                top_logprobs = torch.gather(class_logprobs, -1, predicted_classes)  # (batch_size * num_beams, per_node_beam_size)

            # Here we expand the last log probs to `(batch_size * beam_size,
            # per_node_beam_size)` so that we can add them to the current log
            # probs for this timestep. This lets us maintain the log
            # probability of each element on the beam.
            # shape: (batch_size * beam_size, per_node_beam_size)
            expanded_last_logprobs = (
                last_logprobs.unsqueeze(2)
                .expand(batch_size, self.beam_size, self.per_node_beam_size)
                .reshape(batch_size * self.beam_size, self.per_node_beam_size)
            )
            # shape: (batch_size * beam_size, per_node_beam_size)
            summed_top_logprobs = top_logprobs + expanded_last_logprobs

            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_summed = summed_top_logprobs.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # shape: (batch_size, beam_size * per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(
                batch_size, self.beam_size * self.per_node_beam_size
            )
            # Append the predictions to the current beam.
            reshaped_beam = (
                predictions.view(batch_size * self.beam_size, 1, -1)
                .repeat(1, self.per_node_beam_size, 1)
                .reshape(batch_size, self.beam_size * self.per_node_beam_size, -1)
            )
            # batch_size, (beam_size * per_node_beach_size), #token
            reshaped_beam = torch.cat([reshaped_beam, reshaped_predicted_classes.unsqueeze(-1)], dim=-1)

            # Keep only the top `beam_size` beam indices.
            # shape: (batch_size, beam_size), (batch_size, beam_size)
            restricted_beam_logprobs, restricted_beam_indices = reshaped_summed.topk(
                self.beam_size
            )
            predictions = reshaped_beam.gather(
                1, restricted_beam_indices.unsqueeze(-1).repeat(1,1,reshaped_beam.shape[-1])
            )

            # shape: (batch_size, beam_size)
            last_logprobs = restricted_beam_logprobs

        if not torch.isfinite(last_logprobs).all():
            warnings.warn(
                "Infinite log probs encountered. Some final captions may not "
                "make sense. This can happen when the beam size is larger than"
                " the number of valid (non-zero probability) transitions that "
                "the step function produces.",
                RuntimeWarning,
            )

        # Optionally select best beam and its logprobs.
        if only_return_best:
            # shape: (batch_size, sequence_length)
            predictions = predictions[:, 0, :]
            last_logprobs = last_logprobs[:, 0]
        num_valid = (predictions != self._eos_index).sum(dim=-1)
        num_valid += (predictions == self._eos_index).sum(dim=-1) > 0
        num_valid = num_valid - start_predictions.shape[1]
        num_valid = num_valid.clip(min=1)

        last_logprobs = last_logprobs / num_valid

        return predictions, last_logprobs


class TransformerDecoderTextualHead(TextualHead):
    # used by unifusiondecoder and imageencodertextdecoder pipelines
    def __init__(
        self,
        visual_feature_size: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        attention_heads: int,
        feedforward_size: int,
        dropout: float = 0.1,
        norm_type: str = "post",
        mask_future_positions: bool = True,
        max_caption_length: int = 30,
        padding_idx: int = 0,
        decoder_type=None,
        visual_projection_type=None,
        not_tie_weight=None,
        output_hidden_states=None,
        use_mlp_wrapper=None,
        cosine_linear=False,
    ):
        super().__init__(visual_feature_size, vocab_size, hidden_size)
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        assert mask_future_positions
        self.padding_idx = padding_idx

        if visual_feature_size is not None:
            self.visual_projection = create_projecton_layer(
                visual_projection_type, visual_feature_size, self.textual_feature_size)
        else:
            self.visual_projection = nn.Identity()
        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.textual_feature_size,
            dropout=dropout,
            max_caption_length=max_caption_length,
            padding_idx=padding_idx,
        )
        self.transformer = create_decoder(
            decoder_type=decoder_type,
            norm_type=norm_type,
            textual_feature_size=self.textual_feature_size,
            attention_heads=self.attention_heads,
            feedforward_size=self.feedforward_size,
            dropout=dropout,
            num_layers=self.num_layers,
            output_hidden_states=output_hidden_states,
            use_mlp_wrapper=use_mlp_wrapper,
        )
        self.apply(self._init_weights)

        if cosine_linear:
            assert NotImplementedError
        else:
            # Create an output linear layer and tie the input and output word
            # embeddings to reduce parametejs.
            self.output = nn.Linear(self.textual_feature_size, vocab_size)
        if not not_tie_weight:
            self.output.weight = self.embedding.words.weight

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            #if module.padding_idx is not None:
                #module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        hidden_states,
        caption_tokens,
        hidden_valid_mask=None, # can be None
        caption_lengths=None, # useless
        bi_valid_mask_caption=None,
        #caption_mask=None,
        encoder_history_states=None,
        return_dict=False,
    ):
        if return_dict:
            ret = {}

        projected_visual_features = self.visual_projection(hidden_states) if hidden_states is not None else None
        if return_dict:
            ret['projected_visual_features'] = projected_visual_features
        batch_size, max_caption_length = caption_tokens.size()
        caption_embeddings = self.embedding(caption_tokens)

        # An additive mask for masking the future (one direction).
        uni_mask_zero_neg = self._generate_future_mask(
            max_caption_length, caption_embeddings.dtype, caption_embeddings.device
        )

        # We transpose the first two dimensions of tokens embeddings and visual
        # features, as required by decoder.
        caption_embeddings = caption_embeddings.transpose(0, 1)
        if projected_visual_features is not None:
            projected_visual_features = projected_visual_features.transpose(0, 1)
        else:
            projected_visual_features = torch.zeros(
                (0, caption_embeddings.shape[1], caption_embeddings.shape[2]),
                dtype=caption_embeddings.dtype,
                device=caption_embeddings.device,
            )

        extra_param = {}
        if bi_valid_mask_caption is not None:
            extra_param = {'tgt_bi_valid_mask': bi_valid_mask_caption}
        if not isinstance(self.transformer, torch.nn.modules.transformer.TransformerDecoder):
            extra_param['encoder_history_states'] = encoder_history_states

        # if transformer here is the pytorch/decoder, there is no chance, the
        # output is always tensor
        trans_out = self.transformer(
            caption_embeddings,
            projected_visual_features,
            memory_key_padding_mask=(hidden_valid_mask.logical_not() if hidden_valid_mask is not None else None),
            tgt_mask=uni_mask_zero_neg,
            #tgt_key_padding_mask=caption_mask,
            #encoder_history_states=encoder_history_states,
            **extra_param,
        )
        if isinstance(trans_out, tuple):
            textual_features = trans_out[0]
        else:
            assert isinstance(trans_out, torch.Tensor)
            textual_features = trans_out
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)
        if return_dict:
            ret['textual_features'] = textual_features

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)
        if isinstance(trans_out, tuple):
            if return_dict:
                ret['output_logits'] = output_logits
                ret['history'] = trans_out[1]
                return ret
            else:
                return output_logits, trans_out[1]
        else:
            if return_dict:
                ret['output_logits'] = output_logits
                return ret
            else:
                return output_logits

    def _generate_future_mask(
        self, size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

def convert2valid(shape, length=None, device='cuda'):
    if length is None:
        valid = torch.full(shape, fill_value=True, device=device)
    else:
        ones = torch.ones(shape, device=device)
        valid = ones.cumsum(dim=1) <= length.unsqueeze(1)
    return valid

class SmoothLabelCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1, log_prefix='', ignore_index=None):
        super().__init__()
        self.eps = eps
        self.log_soft = nn.LogSoftmax(dim=1)
        #self.kl = nn.KLDivLoss(reduction='batchmean')
        self.kl = nn.KLDivLoss(reduction='none')

        # for verbose printing only
        #self.register_buffer('iter', torch.tensor(0))
        self.iter = 0
        self.max_loss = 0
        self.min_loss = 0
        self.log_prefix = log_prefix
        self.ignore_index = ignore_index

    def forward(self, feature, target):
        # if it is fp16, convert it to fp32 explicitly as some trainer will not
        # do automatically
        feature = feature.float()
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target[valid_mask]
            feature = feature[valid_mask]
        assert target.numel() > 0
        debug_print = (self.iter % 100) == 0
        self.iter += 1
        eps = self.eps
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(feature)
        if debug_print:
            with torch.no_grad():
                prob = torch.nn.functional.softmax(feature.detach(), dim=1)
                num = feature.size(0)
                avg_prob = prob[torch.arange(num), target].mean()
                logging.info('{}: iter={}, avg pos = {}, max loss = {}, min loss = {}'.format(
                    self.log_prefix,
                    self.iter,
                    avg_prob,
                    self.max_loss,
                    self.min_loss,
                ))
                self.max_loss = 0
                self.min_loss = 10000000
        loss = self.kl(log_prb, one_hot)
        with torch.no_grad():
            if len(loss) > 0:
                self.max_loss = max(self.max_loss, loss.max().cpu())
                self.min_loss = min(self.min_loss, loss.min().cpu())
        return loss.sum(dim=1).mean()

#class ScstRewardCriterion(torch.nn.Module):
    #CIDER_REWARD_WEIGHT = 1

    #def __init__(self, cider_cached_tokens='corpus', baseline_type='greedy'):
        #from .cider.pyciderevalcap.ciderD.ciderD import CiderD
        #self.CiderD_scorer = CiderD(df=cider_cached_tokens)
        #assert baseline_type in ['greedy', 'sample']
        #self.baseline_type = baseline_type
        #self._cur_score = None
        #self._greedy_score = None
        #self._pos_reward = None
        #self._neg_reward = None
        #super().__init__()

    #def forward(self, gt_res, greedy_res, sample_res, sample_logprobs):
        #batch_size = len(gt_res)
        #sample_res_size = len(sample_res)
        #seq_per_img = sample_res_size // batch_size

        #gen_res = []
        #gen_res.extend(sample_res)
        #gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        #if self.baseline_type == 'greedy':
            #assert len(greedy_res) == batch_size
            #gen_res.extend(greedy_res)
            #gt_idx.extend([i for i in range(batch_size)])

        #scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)

        #if self.baseline_type == 'greedy':
            #import numpy as np
            #baseline = scores[-batch_size:][:, np.newaxis]
        #else:
            #sc_ = scores.reshape(batch_size, seq_per_img)
            #baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        ## sample - baseline
        #reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        #with torch.no_grad():
            #self._cur_score = reward.mean()
            #self._greedy_score = scores[sample_res_size:].mean()
        #reward = reward - baseline
        #with torch.no_grad():
            #self._pos_reward = (reward > 0).sum()
            #self._neg_reward = (reward < 0).sum()
        #reward = reward.reshape(sample_res_size)

        #reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        #loss = - sample_logprobs * reward
        #loss = loss.mean()
        #return loss

    #def get_score(self):
        #return self._cur_score

    #def get_info(self):
        #return {
            #'curr_score': self._cur_score,
            #'greedy_score': self._greedy_score,
            #'pos_reward': self._pos_reward,
            #'neg_reward': self._neg_reward,
        #}

    #def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        #'''
        #gen_res: generated captions, list of str
        #gt_idx: list of int, of the same length as gen_res
        #gt_res: ground truth captions, list of list of str.
            #gen_res[i] corresponds to gt_res[gt_idx[i]]
            #Each image can have multiple ground truth captions
        #'''
        #gen_res_size = len(gen_res)

        #from collections import OrderedDict
        #res = OrderedDict()
        #for i in range(gen_res_size):
            #res[i] = [self._wrap_sentence(gen_res[i])]

        #gts = OrderedDict()
        #gt_res_ = [
            #[self._wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))]
                #for i in range(len(gt_res))
        #]
        #for i in range(gen_res_size):
            #gts[i] = gt_res_[gt_idx[i]]

        #res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
        #_, batch_cider_scores = self.CiderD_scorer.compute_score(gts, res_)
        #scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        #return scores

    #def _wrap_sentence(self, s):
        ## ensure the sentence ends with <eos> token
        ## in order to keep consisitent with cider_cached_tokens
        #r = s.strip()
        #if r.endswith('.'):
            #r = r[:-1]
        #r += ' <eos>'
        #return r


class CaptioningModel(nn.Module):
    def __init__(
        self,
        visual,
        textual,
        sos_index=1,
        eos_index=2,
        decoder=None,
        #use_masked_as_input_for_train=False,
        loss_type=None,
        context_not_share_embedding=False,
        scst=False,
        tokenizer=None,
        scst_temperature=1.,
        use_history_for_infer=False,
        pooling_images=None,
        num_image_with_embedding=0,
    ):
        super().__init__()
        self.image_encoder = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx

        # These boundary indices are needed for beam search.
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.decoder = decoder
        self.scst = scst
        self.tokenizer = tokenizer

        if self.scst:
            raise NotImplementedError
            #from .utils_caption_evaluate import (
                    #ScstRewardCriterion)
            #self.scst_criterion = ScstRewardCriterion(
                #cider_cached_tokens='data/coco_caption/gt/coco-train-words.p',
                #baseline_type='greedy',
            #)
            #self.scst_fwd_times = 0
            #self.scst_temperature = scst_temperature
        if loss_type is None:
            self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        elif loss_type == 'smooth':
            self.loss = SmoothLabelCrossEntropyLoss(ignore_index=self.padding_idx)
        else:
            raise NotImplementedError(loss_type)
        #self.use_masked_as_input_for_train = use_masked_as_input_for_train

        self.verbose = {'num_has_image': 0, 'num_no_image': 0}
        self.context_not_share_embedding = context_not_share_embedding
        if context_not_share_embedding:
            self.context_embedding = self.textual.embedding.clone()
            # check whether the parameters are shared or not. it should not
            # share
        self.use_history_for_infer = use_history_for_infer
        self.pooling_images = pooling_images

        if num_image_with_embedding:
            logging.info('creating temperal embedding')
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, self.textual.visual_feature_size)) for _ in range(num_image_with_embedding)
            )
        self.num_image_with_embedding = num_image_with_embedding

    def forward(self, batch):
        result = self.forward_one(batch, return_info=False)
        return result

        # shape: (batch_size, channels, height, width)
    def forward_one(self, batch, return_info=False):
        # shape: (batch_size, max_caption_length, vocab_size)
        if 'image' in batch:
            if isinstance(batch['image'], (list, tuple)):
                features = [self.image_encoder(im) for im in batch['image']]
                if self.num_image_with_embedding:
                    features = [f + e for f, e in zip(features, self.img_temperal_embedding)]
                if self.pooling_images is None:
                    visual_features = torch.cat(features, dim=1)
                elif self.pooling_images == 'avg':
                    visual_features = torch.stack(features, dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
            else:
                visual_features = self.image_encoder(batch['image'])
        else:
            visual_features = None
        visual_features_valid = None
        if 'context' in batch:
            context_embedding = self.context_embedding if self.context_not_share_embedding else self.textual.embedding
            all_context = [visual_features]
            all_valid = [convert2valid(visual_features.shape[:2])]
            for info in batch['context']:
                context = context_embedding(info['tokens'])
                valid = convert2valid(info['tokens'].shape, info['length'])
                all_context.append(context)
                all_valid.append(valid)
            visual_features = torch.cat(all_context, dim=1)
            visual_features_valid = torch.cat(all_valid, dim=1)

        if not self.training or (not self.scst):
            return self.forward_one_ce(batch, visual_features, visual_features_valid, return_info)
        else:
            assert self.training and self.scst
            return self.forward_one_scst(batch, visual_features, visual_features_valid)

    def forward_one_scst(self, batch, visual_features, visual_features_valid):
        self.eval()
        def _ids_to_captions(all_ids):
            captions = []
            for ids in all_ids:
                c = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                captions.append(c)
            return captions
        with torch.no_grad():
            greedy_res = self.infer(batch, visual_features, visual_features_valid)
            greedy_res_raw = greedy_res['predictions']
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
            greedy_res = _ids_to_captions(greedy_res_raw)

        self.train()
        search_param = {
            'do_sample': True,
            #'top_k': 5,
            'top_p': 1,
            'num_return_sequences': 5,
            'temperature': self.scst_temperature,
        }
        infer_res = self.infer(
            batch,
            visual_features,
            visual_features_valid,
            search_param,
        )
        sample_res = _ids_to_captions(infer_res['predictions'])
        gt_res = list(zip(*[[j_th_image_cap for j_th_image_cap in i_th_caption['caption']] for i_th_caption in batch['all_caption']]))
        loss = self.scst_criterion(gt_res, greedy_res, sample_res, infer_res['logprobs'])
        if (self.scst_fwd_times % 100) == 0:
            info = self.scst_criterion.get_info()
            logging.info(pformat(info))
        self.scst_fwd_times += 1
        return {'decoder_loss': loss}

    def forward_one_ce(self, batch, visual_features, visual_features_valid, return_info):
        has_image = (visual_features is not None)
        assert has_image == ('image' in batch)
        if self.training:
            #if self.use_masked_as_input_for_train:
                #caption_token_input = batch["masked_caption_tokens"]
            #else:
            caption_token_input = batch["caption_tokens"]
            #caption_lengths = batch["caption_lengths"]

            output_logits = self.textual(
                visual_features,
                caption_token_input,
                #caption_lengths=caption_lengths,
                hidden_valid_mask=visual_features_valid,
                bi_valid_mask_caption=batch.get('bi_valid_mask_caption'),
            )
            output_dict = {}
            #output_logits = x['output_logits']
            #ipdb> output_logits.shape
            #torch.Size([2, 13, 30522])
            #ipdb> batch['caption_tokens'].shape
            #torch.Size([2, 13])
            if 'need_predict' in batch:
                target = batch["caption_tokens"].clone()
                if self.padding_idx is not None:
                    target[batch['need_predict'] == 0] = self.padding_idx
            else:
                assert ValueError()
                #target = batch["caption_tokens"]
            need_predict = batch['need_predict']
            feat = output_logits[:, :-1].contiguous()
            target = target[:, 1:].contiguous()
            need_predict = need_predict[:, 1:].contiguous()
            feat = feat.view(-1, self.textual.vocab_size)
            target = target.view(-1)
            need_predict = need_predict.view(-1)

            valid_mask = need_predict == 1
            #valid_mask2 = target != self.padding_idx
            #assert (valid_mask.long() - valid_mask2.long()).abs().sum().cpu() == 0

            target = target[valid_mask]
            feat = feat[valid_mask]
            loss = self.loss(feat, target)
            if (self.verbose['num_has_image'] + self.verbose['num_no_image']) % 200 == 0:
                logging.info(self.verbose)
            hint = 'l' if 'context_target_type' not in batch else batch['context_target_type'][0]
            if has_image:
                output_dict.update({'vl_{}_loss'.format(hint): loss})
                self.verbose['num_has_image'] += 1
            else:
                output_dict.update({'l_{}_loss'.format(hint): loss})
                self.verbose['num_no_image'] += 1

            if return_info:
                output_dict['feat'] = feat
        else:
            output_dict = self.infer(batch, visual_features, visual_features_valid)
        return output_dict

    def infer(self, batch, visual_features, visual_features_valid,
              search_param=None):
        batch_size = visual_features.size(0)
        if 'prefix' not in batch:
            start_predictions = visual_features.new_full(
                (batch_size,1), self.sos_index
            ).long()
        else:
            # if batch size is larger than 1, the prefix length could be
            # different, and we have to padding non-valid data, which
            # is not supported
            assert len(batch['prefix']) == 1, 'not supported'
            start_predictions = batch['prefix'].long()

        self.prev_encoded_layers = None
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        decoding_step = functools.partial(
            self.decoding_step, visual_features, visual_features_valid,
            batch.get('bi_valid_mask_caption')
        )

        search_param = search_param or {}
        # the start_predictions are not in predicted_caption
        predicted_caption, logprobs = self.decoder.search(
            start_predictions, decoding_step, **search_param
        )
        if 'prefix' in batch:
            # we need to remove prefix from predicted_caption
            predicted_caption = predicted_caption[:, start_predictions.shape[1]:]
        output_dict = {
            'predictions': predicted_caption,
            'logprobs': logprobs,
        }
        return output_dict

    def decoding_step(
        self, visual_features, visual_features_valid, bi_valid_mask_caption, partial_captions
    ):
        # Expand and repeat image features while doing beam search.
        batch_size = visual_features.shape[0]
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            batch_size, num_token, channels = visual_features.size()
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, num_token, channels
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a timestep. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        logits = self.textual(
            visual_features,
            partial_captions,
            caption_lengths=caption_lengths,
            hidden_valid_mask=visual_features_valid,
            bi_valid_mask_caption=bi_valid_mask_caption,
            encoder_history_states=self.prev_encoded_layers,
        )
        if self.scst or self.use_history_for_infer:
            if isinstance(logits, tuple) and len(logits) == 2:
                if self.prev_encoded_layers is None:
                    self.prev_encoded_layers = logits[1]
                else:
                    self.prev_encoded_layers = [torch.cat((p, c), dim=1) for p, c in
                                                zip(self.prev_encoded_layers, logits[1])]
                #self.prev_encoded_layers = None
                logits = logits[0]
        return logits[:, -1, :].float()

class GeneratorWithBeamSearch(object):
    def __init__(
        self,
        eos_index: int,
        max_steps: int,
        beam_size: int,
        per_node_beam_size: int = 2,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        temperature: float = 1,
    ) -> None:
        self._eos_index = eos_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        # NOTE: Expand >1 words to leave some spare tokens to keep the
        # beam size, because some sentences may end here and cannot expand
        # in the next level
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

        assert self.per_node_beam_size > 1
        assert self.length_penalty > 0, "`length_penalty` should be strictely positive."
        assert self.repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert self.temperature > 0, "`temperature` should be strictely positive."

    def search(
        self,
        input_ids,
        step,
        num_keep_best= 1,
        do_sample=False,
        top_k=None,
        top_p=None,
        num_return_sequences=1,
    ):
        if num_return_sequences != 1:
            input_ids = input_ids[:, None, :].expand(
                input_ids.shape[0], num_return_sequences, input_ids.shape[1])
            input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        batch_size, cur_len = input_ids.shape
        num_beams = self.beam_size
        pad_token_id = self._eos_index
        eos_token_ids = [self._eos_index]
        per_node_beam_size = self.per_node_beam_size
        repetition_penalty = self.repetition_penalty
        temperature = self.temperature

        # Expand input to num beams
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

        #prefix_len = cur_len
        #max_length = self.max_steps + prefix_len
        max_length = self.max_steps
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_keep_best, max_length, self.length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        ## cache compute states
        #past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            scores = step(input_ids)  # (batch_size * num_beams, cur_len, vocab_size)
            vocab_size = scores.shape[-1]

            ## if model has past, then set the past variable to speed up decoding
            #if self._do_output_past(outputs):
                #past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample [per_node_beam_size] next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1),
                        num_samples=per_node_beam_size)  # (batch_size * num_beams, TOPN_PER_BEAM)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, per_node_beam_size)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, per_node_beam_size)
                # Match shape of greedy beam search
                beam_indices = torch.arange(num_beams, device=next_words.device) * vocab_size
                beam_indices = beam_indices.repeat(batch_size, per_node_beam_size)
                next_words = next_words.view(batch_size, per_node_beam_size * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
                next_words = next_words + beam_indices
                next_scores = next_scores.view(batch_size, per_node_beam_size * num_beams)  # (batch_size, TOPN_PER_BEAM * num_beams)
            else:
                # do greedy beam search
                scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
                next_scores, next_words = torch.topk(_scores, per_node_beam_size * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_words.size() == (batch_size, per_node_beam_size * num_beams)

            # next batch beam content
            # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for batch_ex in range(batch_size):

                # if we are done with this sentence
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []
                # next words for this sentence
                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                    # get beam and word IDs
                    beam_id = idx // vocab_size
                    word_id = idx % vocab_size

                    # end of sentence, or next word
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                        )
                    else:
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # update next beam content
                if cur_len + 1 == max_length:
                    assert len(next_sent_beam) == 0
                else:
                    assert len(next_sent_beam) == num_beams

                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # re-order internal states
            #if past:
                #reordered_past = []
                #for layer_past in past:
                    ## get the correct batch idx from layer past batch dim
                    ## batch dim of `past` and `mems` is at 1st position
                    #reordered_layer_past = [layer_past[i].unsqueeze(0).clone().detach() for i in beam_idx]
                    #reordered_layer_past = torch.cat(reordered_layer_past, dim=0)
                    ## check that shape matches
                    #assert reordered_layer_past.shape == layer_past.shape
                    #reordered_past.append(reordered_layer_past)
                #past = tuple(reordered_past)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(batch_size):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # select the best hypotheses
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best,
                dtype=torch.float).fill_(-1e5).to(input_ids.device)
        all_best = []

        for i, hypotheses in enumerate(generated_hyps):
            best = []
            hyp_scores = torch.tensor([x[0] for x in hypotheses.hyp])
            _, best_indices = torch.topk(hyp_scores,
                    min(num_keep_best, len(hyp_scores)), largest=True)
            for best_idx, hyp_idx in enumerate(best_indices):
                conf, best_hyp = hypotheses.hyp[hyp_idx]
                best.append(best_hyp)
                logprobs[i, best_idx] = conf
                tgt_len[i, best_idx] = len(best_hyp) + 1  # +1 for the <EOS> symbol

            all_best.append(best)

        # generate target batch, pad to the same length
        decoded = input_ids.new(batch_size, num_keep_best, max_length).fill_(pad_token_id)
        for batch_idx, best in enumerate(all_best):
            for best_idx, hypo in enumerate(best):
                decoded[batch_idx, best_idx, : tgt_len[batch_idx, best_idx] - 1] = hypo
                decoded[batch_idx, best_idx, tgt_len[batch_idx, best_idx] - 1] = eos_token_ids[0]
        if num_keep_best == 1:
            decoded = decoded.squeeze(dim=1)
        return decoded, logprobs

class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def _length_norm(self, length):
        #return length ** self.length_penalty
        # beam search alpha: https://opennmt.net/OpenNMT/translation/beam_search/
        return (5 + length) ** self.length_penalty / (5 + 1) ** self.length_penalty

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        #score = sum_logprobs / len(hyp) ** self.length_penalty
        score = sum_logprobs / self._length_norm(len(hyp))
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            #return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
            return self.worst_score >= best_sum_logprobs / self._length_norm(self.max_length)

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

