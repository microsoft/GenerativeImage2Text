from torch.nn import functional as F
import torch
from collections import defaultdict


def get_trie(tokenizer):
    output_tokens = get_output_vocab_tokens(tokenizer)
    from .trie_decoder import TokenTrie
    ret = TokenTrie.construct(output_tokens)
    return ret

def get_trie_vocab_texts():
    # please create this image net file based on readme or replace it with a
    # new file name
    fname = './aux_data/imagenet/imagenet_unique_readable_names.txt'
    with open(fname, 'r') as fp:
        return list(fp)

def get_output_vocab_tokens(tokenizer):
    answermap = get_trie_vocab_texts()
    ret = []
    for a in answermap:
        token = tokenizer(a, padding='do_not_pad', add_special_tokens=False)
        ret.append(token['input_ids'] + [tokenizer.sep_token_id])
    return ret

class TrieAutoRegressiveBeamSearch(object):
    def __init__(
        self,
        eos_index: int,
        max_steps: int = 50,
        beam_size: int = 5,
        trie=None,
    ):
        self._eos_index = eos_index
        self.max_steps = max_steps
        assert beam_size == 1
        self.beam_size = beam_size
        self.per_node_beam_size = 1
        self.trie = trie

    def search(self, start_predictions, step,
               only_return_best=True,
               do_sample=False,
               top_k=0,
               top_p=None,
               num_return_sequences=1,
               temperature=1,
               ):
        self.trie.reset()
        if num_return_sequences > 1:
            start_predictions = start_predictions[:, None, :].expand(
                start_predictions.shape[0],
                num_return_sequences,
                start_predictions.shape[1])
            start_predictions = start_predictions.reshape(-1, start_predictions.shape[-1])

        batch_size = start_predictions.size()[0]
        predictions = start_predictions.unsqueeze(1).expand((batch_size, self.beam_size, start_predictions.shape[-1]))
        start_class_logits = step(start_predictions)
        start_class_logprobs = F.log_softmax(start_class_logits, dim=1)
        idx = self.trie.get_curr_valid()
        start_class_logprobs[0, idx] += start_class_logits.max() - start_class_logits.min() + 1

        num_classes = start_class_logprobs.size()[1]

        # shape: (batch_size, beam_size), (batch_size, beam_size)
        start_top_logprobs, start_predicted_classes = start_class_logprobs.topk(
            self.beam_size
        )
        self.trie.move(start_predicted_classes[0, 0].item())

        if (
            self.beam_size == 1
            and (start_predicted_classes == self._eos_index).all()
        ):
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
            idx = self.trie.get_curr_valid()
            class_logprobs[0, idx] += class_logits.max() - class_logits.min() + 1

            # Set logprobs of last predicted tokens as high negative value to avoid
            # repetition in caption.
            #class_logprobs = class_logprobs.scatter(1, predictions_so_far[:, -1].view((-1, 1)), -10000)

            # shape (both): (batch_size * beam_size, per_node_beam_size)
            top_logprobs, predicted_classes = class_logprobs.topk(
                self.per_node_beam_size
            )
            self.trie.move(predicted_classes[0, 0].item())

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
            pass

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

class TokenNode():
    def __init__(self):
        self.children = defaultdict(TokenNode)

class TokenTrie(object):
    def __init__(self):
        self.root = TokenNode()
        self.curr = self.root

    @classmethod
    def construct(self, all_tokens):
        ret = TokenTrie()
        for ts in all_tokens:
            ret.insert(ts)
        return ret

    def insert(self, tokens):
        cur = self.root
        for t in tokens:
            cur = cur.children[t]

    def get_valid(self, tokens):
        r = self.root
        for t in tokens:
            r = r.children.get(t)
            if r is None:
                return []
        return list(r.children.keys())

    def reset(self):
        self.curr = self.root

    def get_curr_valid(self):
        return list(self.curr.children.keys())

    def move(self, t):
        assert t in self.curr.children
        self.curr = self.curr.children[t]

