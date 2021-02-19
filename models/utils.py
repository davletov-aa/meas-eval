from transformers import BertConfig
from transformers import RobertaConfig
from transformers import XLNetConfig
from torch.utils.data import DataLoader, TensorDataset
from .roberta_qa import RobertaForQA
import torch
import os
import json
from collections import Counter
import pandas as pd
from glob import glob
import numpy as np
from collections import namedtuple


class DataProcessor:
    def __init__(self, train_docs_dir, split_qualifier=True, add_double_sentences=True):
        self.split_qualifier = split_qualifier
        self.add_double_sentences = add_double_sentences
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
        from nltk.tokenize import RegexpTokenizer

        train_texts = ' '.join([open(txt_file, 'r').read() for txt_file in glob(os.path.join(train_docs_dir,'*.txt'))])
        trainer = PunktTrainer()
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.train(train_texts)

        self.sentenizer = PunktSentenceTokenizer(trainer.get_params())
        self.tokenizer = RegexpTokenizer(r'\w+|\(|\)|\[|\]|[-–.,]|\S+')


    def get_dataframes(self, docs_dir, annotations_dir=None, part='train'):

        from collections import namedtuple
        Sentence = namedtuple('Sentence', ['start', 'end', 'text'])

        tsv_files = glob(os.path.join(annotations_dir,'*.tsv')) if annotations_dir is not None else []
        txt_files = glob(os.path.join(docs_dir,'*.txt'))

        docs = {'docId': [], 'text': []}
        for txt_file in txt_files:
            docs['text'].append(open(txt_file, 'r').read())
            docs['docId'].append(txt_file.split('/')[-1].split('.')[0])
        docs = pd.DataFrame(docs)

        docs.loc[:, 'texts'] = docs.text.apply(
            lambda x: [
                Sentence(start, end, text) for (start, end), text in zip(
                    self.sentenizer.span_tokenize(x), self.sentenizer.tokenize(x)
                )
            ]
        )
        if self.add_double_sentences and not part != 'train':
            docs.loc[:, 'texts'] = docs.apply(
                lambda r: r.texts + [
                    Sentence(
                        r.texts[i].start,
                        r.texts[i + 1].end,
                        r.text[r.texts[i].start:r.texts[i + 1].end]
                    ) for i in range(len(r.texts) - 1)
                ],
                 axis=1
            )

        docs = (docs
            .set_index(['docId', 'text'])['texts']
            .apply(pd.Series).stack().reset_index()
            .drop(['level_2', 'text'], axis=1)
            .rename(columns={0: 'text'})
        )

        if annotations_dir is not None:

            annotations = pd.read_csv(tsv_files[0], sep='\t')
            for tsv_file in tsv_files[1:]:
                annotations = annotations.append(pd.read_csv(tsv_file, sep='\t'), ignore_index=True)
            annotations = annotations.replace(np.nan, '{}')
            annotations.loc[:, 'other'] = annotations.other.apply(lambda x: eval(x))
        else:
            annotations = pd.DataFrame(columns=['docId', 'annotSet', 'annotType', 'startOffset', 'endOffset', 'annotId', 'text', 'other'])
        if self.split_qualifier:
            corrQualifiers = {}
            for row in annotations[annotations.annotType == 'Qualifier'].itertuples():
                docId, start, end, qualifies = row.docId, row.startOffset, row.endOffset, row.other['Qualifies']
                annotType = annotations[(annotations.docId == docId) & (annotations.annotId == qualifies)].annotType.values[0]
                corrQualifiers[(start, end, docId)] = f'{annotType}Qualifier'
            annotations.loc[:, 'annotType'] = annotations.apply(lambda r: r.annotType if r.annotType != 'Qualifier' else corrQualifiers[(r.startOffset, r.endOffset, r.docId)], axis=1)
        return docs, annotations

    def get_ner_examples(
        self, docs_dir,
        annotations_dir=None,
        only_labels='Quantity+MeasuredProperty+MeasuredEntity+Qualifier'
    ):

        sents, annotations = self.get_dataframes(docs_dir, annotations_dir)

        from collections import namedtuple
        Token = namedtuple('Token', ['start', 'end', 'text'])
        AnnotatedToken = namedtuple('AnnotatedToken', ['start', 'end', 'text', 'label'])

        examples = []
        num_ = 0
        def retrieve_labels_in_sentence(sent, doc_annots):
            AnnotatedSpan = namedtuple('AnnotatedSpan', ['start', 'end', 'label'])
            df = doc_annots[(doc_annots.startOffset >= sent.start) & (doc_annots.endOffset <= sent.end)]
            return [AnnotatedSpan(r.startOffset, r.endOffset, r.annotType) for r in df.itertuples()]

        for sent in sents.itertuples():
            sent_annotations = annotations[(annotations.docId == sent.docId) & (annotations.annotType.isin(only_labels.split('+')))]
            tokens = [
                Token(sent.text.start + start, sent.text.start + end, token)
                for (start, end), token in zip(
                    self.tokenizer.span_tokenize(sent.text.text),
                    self.tokenizer.tokenize(sent.text.text)
                )
            ]
            new_sent = True
            doc_labels = retrieve_labels_in_sentence(sent.text, sent_annotations)
            annotated_tokens = []
            for token in tokens:
                labels = [
                    (
                        label,
                        set(range(token.start, token.end)).intersection(set(range(label.start, label.end)))
                    )
                    for label in doc_labels
                ]
                labels = [(label, positions) for label, positions in labels if positions]

                if len(set([f'{x.label}+{str(x.start)}+{(x.end)}' for (x, y) in labels])) > 1:
                    if new_sent: print(sent.docId, set([f'{x.label}+{str(x.start)}+{(x.end)}' for (x, y) in labels]))
                    num_ += 1 if new_sent else 0
                    new_sent = False

                if labels:
                    annotated_tokens.append(AnnotatedToken(token.start, token.end, token.text, '+'.join(set(label[0].label for label in labels))))
                else:
                    annotated_tokens.append(AnnotatedToken(token.start, token.end, token.text, 'O'))
            if len(annotated_tokens) == len(tokens):
                example = {
                    'annotations': annotated_tokens,
                    'docId': sent.docId
                }
            examples.append(example)
        print(num_)
        return examples

    def get_quantity_bio_examples(
        self, docs_dir,
        output_file=None,
        annotations_dir=None
    ):
        examples = self.get_ner_examples(docs_dir, annotations_dir, 'Quantity')
        if output_file is None:
            return examples
        prev_docId = 'O'
        with open(output_file, 'w') as f:
            for example in examples:
                prev_label = 'O'

                if example['docId'] != prev_docId:
                    print('-DOCSTART-', file=f)
                    print(file=f)

                prev_docId = example['docId']

                for token in example['annotations']:
                    if token.label == 'O':
                        label = 'O'
                    elif prev_label == 'O':
                        label = f'B-{token.label}'
                    else:
                        label = f'I-{token.label}'

                    print(token.text, token.start, token.end, example['docId'], label, sep=' ', file=f)
                    prev_label = token.label
                print(file=f)
        return examples

    def get_quantity_bio_predictions_as_dataframe(self, docs_dir, predictions_file):
        examples = self.get_quantity_bio_examples(docs_dir)
        predictions = [line.rstrip() for line in open(predictions_file).readlines()]
        offset = 0
        result = []
        from itertools import groupby
        def get_bio_spans(predicted_labels):
            start_pos, spans = 0, []
            for key, group in  groupby(predicted_labels):
                group_len = len(list(group))
                if key != 'O':
                    spans.append((start_pos, start_pos + group_len - 1))
                start_pos += group_len
            return spans

        doc_files = glob(os.path.join(docs_dir,'*.txt'))

        docs = {}
        for doc_file in doc_files:
            docs[doc_file.split('/')[-1].split('.')[0]] = open(doc_file, 'r').read()

        setId = 1
        for i, example in enumerate(examples):
            ex_pred_labels = [token.split(' ')[-1] for token in predictions[offset:offset+len(example['annotations'])]]
            for j, (start, end) in enumerate(get_bio_spans(ex_pred_labels)):
                startOffset = example['annotations'][start].start
                endOffset = example['annotations'][end].end
                row = {
                    'docId': example['docId'],
                    'annotSet': setId,
                    'annotType': 'Quantity',
                    'startOffset': startOffset,
                    'endOffset': endOffset,
                    'annotId': f'T1-{setId}',
                    'text': docs[example['docId']][startOffset:endOffset],
                    'other': {}
                }
                result.append(row)
                setId += 1
            offset += len(example['annotations'])
        result = pd.DataFrame(result).drop_duplicates(subset=['docId', 'startOffset', 'endOffset'])
        return result

    def tokenize_sent_with_spans(self, sent):
        from collections import namedtuple
        Token = namedtuple('Token', ['start', 'end', 'text'])
        tokens = [
            Token(sent.start + start, sent.start + end, token)
            for (start, end), token in zip(
                self.tokenizer.span_tokenize(sent.text),
                self.tokenizer.tokenize(sent.text)
            )
        ]
        return tokens
    
    def add_annotations_to_tokens(self, tokens, labels):
        AnnotatedToken = namedtuple('AnnotatedToken', ['start', 'end', 'text', 'label'])
        return [AnnotatedToken(tok.start, tok.end, tok.text, label) for tok, label in zip(tokens, labels)]
    
    def create_qa_examples(self, docs_dir, annots_dir, part='train'):
        docs, annots = self.get_dataframes(docs_dir, annots_dir, part=part)
        docs = docs[docs.docId.isin(annots.docId.unique())]
        Example = namedtuple('Example', ['docId', 'tokens', 'mods', 'sent'])

        examples = []
        if self.split_qualifier:
            spans_types = ['MeasuredEntityQualifier', 'MeasuredPropertyQualifier', 'QuantityQualifier', 'MeasuredEntity', 'MeasuredProperty', 'Quantity']
        else:
            spans_types = ['Qualifier', 'MeasuredEntity', 'MeasuredProperty', 'Quantity']

        def token_in_span(token, start, end):
            if (start <= token.start < token.end <= end or
                start <= token.start < end <= token.end or
                token.start <= start < token.end <= end or
                token.start <= start <= end <= token.end
            ):
                return True
            else:
                return False

        for docId in annots.docId.unique():
            doc_annots = annots[annots.docId == docId]
            doc_texts = docs[docs.docId == docId]
            for annotSet in doc_annots.annotSet.unique():
                quant_annots = doc_annots[doc_annots.annotSet == annotSet]
                other = quant_annots[quant_annots.annotType == 'Quantity'].other.values[0]
                quant_start = quant_annots[quant_annots.annotType == 'Quantity'].startOffset.min()
                quant_end = quant_annots[quant_annots.annotType == 'Quantity'].endOffset.max()

                contexts = doc_texts[doc_texts.text.apply(lambda x: x.start <= quant_start <= quant_end <= x.end)]
                for sent in contexts.itertuples():
                    tokens = self.tokenize_sent_with_spans(sent.text)
                    example = {}
                    labels = {}

                    for annotType in spans_types:
                        if annotType == 'Qualifier':
                            annot_start = quant_annots[quant_annots.annotType == annotType].startOffset.values[0]
                            annot_end = quant_annots[quant_annots.annotType == annotType].endOffset.values[0]
                        else:
                            annot_start = quant_annots[quant_annots.annotType == annotType].startOffset.min()
                            annot_end = quant_annots[quant_annots.annotType == annotType].endOffset.max()
                        if annotType not in quant_annots.annotType.values:
                            labels[annotType] = ['O'] * len(tokens)
                        else:
                            labels[annotType] = [annotType if token_in_span(token, annot_start, annot_end) else 'O' for token in tokens]
                    if 'mods' in other:
                        example['mods'] = '+'.join(other['mods'])
                    else:
                        example['mods'] = 'O'

                    if 'unit' in other:
                        labels['Unit'] = ['Unit' if token.text in other['unit'] and token_in_span(token, quant_start, quant_end) else 'O' for token in tokens]
                    else:
                        labels['Unit'] = ['O'] * len(tokens)

                    if set(labels['Quantity']) == {'O'}:
                        continue

                    example['tokens'] = self.add_annotations_to_tokens(tokens, ['+'.join(labs) for labs in zip(*[labels[annotType] for annotType in spans_types + ['Unit']])])
                    example['docId'] = docId
                    examples.append(
                        Example(
                            example['docId'],
                            example['tokens'],
                            example['mods'],
                            sent.text
                        )
                    )

        return examples


def get_dataloader_and_tensors(
        features: list,
        batch_size: int
):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long
    )
    clfs_labels = torch.tensor(
        [f.clfs_labels for f in features],
        dtype=torch.long
    )
    quant_labels = torch.tensor(
        [f.quant_labels for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, token_type_ids,
        clfs_labels, quant_labels
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader, clfs_labels, quant_labels


models = {
    # "bert-large-uncased": XLNetForQA,
    "roberta-base": RobertaForQA,
    "roberta-large": RobertaForQA
    # "xlnet-large-cased": BertForQA
}

configs = {
    "bert-large-uncased": BertConfig,
    "roberta-large": RobertaConfig,
    "roberta-base": RobertaConfig,
    "xlnet-large-cased": XLNetConfig
}
