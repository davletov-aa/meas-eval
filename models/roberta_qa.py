from transformers import RobertaModel, RobertaConfig
from transformers import BertPreTrainedModel
from transformers import RobertaTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch
from collections import defaultdict


class MeasFeature:
    def __init__(self, input_ids, input_mask, token_type_ids, clfs_labels, quant_labels, example, subtok_pos_to_orig_tok_pos):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.clfs_labels = clfs_labels
        self.quant_labels = quant_labels
        self.example = example
        self.subtok_pos_to_orig_tok_pos = subtok_pos_to_orig_tok_pos


class RobertaForQA(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(
            self,
            config: RobertaConfig,
            args,
            data_processor
        ):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args
        clfs = sorted(args.clfs)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        self.clf2ncls = [2 for clf in clfs]
        if args.concat_quantity_embeddings:
            self.clfs = nn.Linear(config.hidden_size * 2, sum(self.clf2ncls))
            self.quant_clf = nn.Linear(config.hidden_size * 2, self.args.quant_ncls)
        else:
            self.clfs = nn.Linear(config.hidden_size, sum(self.clf2ncls))
            self.quant_clf = nn.Linear(config.hidden_size, self.args.quant_ncls)
        self.clfs_weights = torch.nn.parameter.Parameter(torch.ones(len(clfs) + 1, dtype=torch.float32), requires_grad=True)
        print(self.clfs_weights)

        self.data_processor = data_processor

        self.QUANT_START = '•'
        self.QUANT_END = '⁄'

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            input_labels=None,
            weighting_mode='softmax'
    ):
        assert weighting_mode in ['softmax', 'rsqr+log', 'equal'], f'wrong weighting_mode: {weighting_mode}'
        loss = defaultdict(float)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        clfs = sorted(self.args.clfs)
        sequences_output = self.dropout(outputs[0])  # bs x seq x hidden
        weights = self.clfs_weights

        labels = input_labels['labels']  # bs x all_ncls
        quant_labels = input_labels['quant_labels']  # bs x quant_ncls

        clfs_labels = torch.split(labels, self.clf2ncls, dim=-1)  # len(clfs) x bs x clf_ncls

        if self.args.concat_quantity_embeddings:
            pos = [i for i, clf in enumerate(clfs) if clf == 'Quantity'][0]
            sequences_output = self.concat_quantity_embeddings(sequences_output, clfs_labels[pos])

        sequences_logits = self.clfs(sequences_output)  # bs x seq x all_ncls
        quant_logits = self.quant_clf(sequences_output[:, 0, :])  # bs x quant_ncls
        clfs_logits = torch.split(sequences_logits, self.clf2ncls, dim=-1) # len(clfs) x bs x seq x clf_ncls

        if input_labels is not None:
            ignore_index = sequences_logits.size(1)
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            quant_loss_fct = BCEWithLogitsLoss()
            labels.clamp_(0, ignore_index)

            for clf_id, (clf_logits, clf_labels) in enumerate(zip(clfs_logits, clfs_labels)):
                clf_loss = 0.0
                for clf_logit, clf_label in zip(torch.split(clf_logits, 1, dim=-1), torch.split(clf_labels, 1, dim=1)):
                    clf_logit = clf_logit.squeeze(-1)  # bs x seq
                    clf_label = clf_label.squeeze(-1)  # bs
                    clf_loss += 1.0 / clf_labels.size(-1) * loss_fct(clf_logit, clf_label)
                loss[clfs[clf_id]] = clf_loss

                if weighting_mode == 'softmax':
                    loss['total'] += torch.exp(weights[clf_id]) * clf_loss
                elif weighting_mode == 'rsqr+log':
                    loss['total'] += 1. / (weights[clf_id] ** 2) * clf_loss + torch.log(weights[clf_id])
                elif weighting_mode == 'equal':
                    loss['total'] += 1. / (len(clfs) + 1) * clf_loss
            quant_labels = quant_labels.float()
            quant_loss = quant_loss_fct(quant_logits, quant_labels)
            loss['mods'] = quant_loss
            if weighting_mode == 'softmax':
                loss['total'] += torch.exp(weights[-1]) * quant_loss
            elif weighting_mode == 'rsqr+log':
                loss['total'] += 1. / (weights[-1] ** 2) * quant_loss + torch.log(weights[-1])
            elif weighting_mode == 'equal':
                loss['total'] += 1. / (len(clfs) + 1) * quant_loss

            if weighting_mode == 'softmax':
                loss['total'] /= torch.sum(torch.exp(weights))

        outputs = (loss, sequences_logits, quant_logits)

        return outputs

    def concat_quantity_embeddings(self, hidden_states, quant_labels):
        pool_type = self.args.pool_type
        bs, seq, hs = hidden_states.size()
        device = torch.device('cuda') if self.args.use_cuda else torch.device('cpu')
        output = torch.zeros(bs, seq, 2 * hs, dtype=torch.float32, device=device)
        output[:, :, :hs] = hidden_states
        for ex_id in range(bs):
            start, end = quant_labels[ex_id, 0].item(), quant_labels[ex_id, 1].item()
            if end != start:
                quant_emb = 0.0
                if pool_type == 'mean':
                    quant_emb = hidden_states[ex_id, start:end+1].mean(dim=0)
                elif pool_type == 'max':
                    quant_emb, _ = hidden_states[ex_id, start:end+1].max(dim=0)
                elif pool_type == 'first':
                    quant_emb = hidden_states[ex_id, start]
                else:
                    raise ValueError(f'wrong pool_type: {pool_type}')
                output[:, :, hs:] = quant_emb
            else:
                output[:, :, hs:] = hidden_states[ex_id, start]
        return output


    def convert_dataset_to_features(
            self, docs, annotations, logger, add_question='Find measured entities and properties of marked quantity', part='train'
    ):
        """Loads a data file into a list of `InputBatch`s."""
        # add_question = ''
        clfs = sorted(self.args.clfs)
        if add_question:
            question_tokens = self.tokenizer.tokenize(add_question)
            logger.info(f'question_tokens: {question_tokens}')

        features = []
        num_shown_examples = 0
        max_seq_len = self.args.max_seq_len
        num_too_long_examples = 0

        quant_lab_to_id = {lab: i for i, lab in enumerate(sorted(self.args.quant_classes))}
        quant_id_to_lab = {i: lab for i, lab in enumerate(sorted(self.args.quant_classes))}
        logger.info(f'quant_lab_to_id: {quant_lab_to_id}')

        examples = self.data_processor.create_qa_examples(docs, annotations, part=part)

        for (ex_index, example) in enumerate(examples):

            tokens = [self.tokenizer.cls_token]
            if add_question:
                tokens += question_tokens + [self.tokenizer.sep_token]

            quant_labels = [1 if quant_lab in example.mods.split('+') else 0 for quant_lab in sorted(self.args.quant_classes)]

            clfs_labels = [0] * (2 * len(clfs))
            clfs_labels_starts = {}
            clfs_labels_ends = {}
            subtok_pos_to_orig_tok_pos = {}
            quant_start_added = False
            quant_end_added = False

            for tok_id, (token, subtokens) in enumerate(zip(example.tokens, [self.tokenizer.tokenize(token.text) for token in example.tokens])):
                token_labels = set(token.label.split('+')).difference({'O'})
                if quant_start_added and not quant_end_added and 'Quantity' not in token_labels:
                    tokens.append(self.QUANT_END)
                    quant_end_added = True
                if 'Quantity' in token_labels and not quant_start_added:
                    tokens.append(self.QUANT_START)
                    quant_start_added = True

                offset = len(tokens)
                subtokens_len = len(subtokens)
                subtok_pos_to_orig_tok_pos[(offset, offset + subtokens_len - 1)] = tok_id

                for token_label in token_labels:
                    if token_label not in clfs_labels_starts:
                        clfs_labels_starts[token_label] = offset
                    if token_label not in clfs_labels_ends:
                        if tok_id == len(example.tokens) - 1:
                            clfs_labels_ends[token_label] = offset + subtokens_len - 1
                            continue
                        if token_label not in set(example.tokens[tok_id + 1].label.split('+')):
                            clfs_labels_ends[token_label] = offset + subtokens_len - 1

                tokens.extend(subtokens)

                if tok_id == len(example.tokens) - 1 and quant_start_added and not quant_end_added and 'Quantity' in token_labels:
                    tokens.append(self.QUANT_END)
                    quant_end_added = True

            tokens += [self.tokenizer.sep_token]

            assert quant_start_added and quant_end_added, tokens

            for clf in clfs:
                if clf not in clfs_labels_starts:
                    clfs_labels_starts[clf] = 0
                if clf not in clfs_labels_ends:
                    clfs_labels_ends[clf] = 0
                assert clfs_labels_starts[clf] <= clfs_labels_ends[clf]

            for clf_id, clf in enumerate(clfs):
                clfs_labels[clf_id * 2] = clfs_labels_starts[clf]
                clfs_labels[clf_id * 2 + 1] = clfs_labels_ends[clf]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                num_too_long_examples += 1

            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * max_seq_len
            padding = [self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]] * (max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += [0] * len(padding)

            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            if ex_index < 10:
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("subtokens: %s" % " ".join(
                    [x for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(f"quant_labels: {' '.join([quant_id_to_lab[i] for i, x in enumerate(quant_labels) if x])}")
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                for clf_id, clf in enumerate(clfs):
                    start, end = clfs_labels[clf_id * 2: clf_id * 2 + 2]
                    if start and end:
                        logger.info(f"{clf}: {' '.join(tokens[start:end+1])}")

            features.append(
                MeasFeature(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    clfs_labels=clfs_labels,
                    quant_labels=quant_labels,
                    example=example,
                    subtok_pos_to_orig_tok_pos=subtok_pos_to_orig_tok_pos
                )
            )
        logger.info(f"too long examples proportion: {100.0 * num_too_long_examples / len(features)}%")
        return features
