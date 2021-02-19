import argparse
import logging
import os
import random
import time
import json
from datetime import datetime
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from torch.nn import CrossEntropyLoss

from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from transformers.file_utils import (
    PYTORCH_PRETRAINED_BERT_CACHE,
    WEIGHTS_NAME, CONFIG_NAME
)

from tqdm import tqdm
from models.utils import (
    get_dataloader_and_tensors,
    models, configs, DataProcessor
)
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report
)
from torch.nn import CrossEntropyLoss

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
eval_logger = logging.getLogger("__scores__")
test_logger = logging.getLogger("__test__")


def predict(
        model, eval_dataloader, output_dir,
        eval_fearures, args,
        cur_train_mean_loss=None,
        logger=None, compute_metrics=True,
        eval_script_path='../MeasEval/eval/measeval-eval.py'
    ):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    clfs = sorted(args.clfs)
    device = torch.device('cuda') if not args.no_cuda else torch.device('cpu')

    metrics = defaultdict(float)
    nb_eval_steps = 0

    clfs_preds = []
    quant_preds = []

    for batch_id, batch in enumerate(tqdm(
            eval_dataloader, total=len(eval_dataloader),
            desc='validation ... '
        )):

        batch = tuple([elem.to(device) for elem in batch])
        input_ids, input_mask, token_type_ids, b_clfs_labels, b_quant_labels = batch

        with torch.no_grad():
            loss, clfs_logits, quant_logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=input_mask,
                input_labels={
                    'labels': b_clfs_labels,
                    'quant_labels': b_quant_labels
                },
                weighting_mode=args.weighting_mode
            )

        if compute_metrics:
            for key, value in loss.items():
                metrics[f'eval_{key}_loss'] += value.mean().item()

        nb_eval_steps += 1
        clfs_preds.append(clfs_logits.detach().cpu().numpy())
        quant_preds.append(quant_logits.detach().cpu().numpy())

    clfs_preds = np.concatenate(clfs_preds, axis=0)  # n_examples x seq x all_ncls
    quant_preds = np.concatenate(quant_preds, axis=0)  # n_examples x quant_ncls

    clfs_scores = softmax(clfs_preds, axis=1).max(axis=-1)  # n_examples x seq x all_ncls
    quant_scores = 1. / (1 + np.exp(-quant_preds))  # n_examples x quant_ncls

    clfs_preds = np.argmax(clfs_preds, axis=1)  # n_examples x all_ncls

    predictions = defaultdict(list)
    quant_id_to_labs = {i: lab for i, lab in enumerate(sorted(args.quant_classes))}
    annotSet = 1
    T = {'Quantity': 'T1', 'MeasuredProperty': 'T2', 'MeasuredEntity': 'T3'}
    for ex_id, (ex_feature, ex_clfs_preds, ex_quant_preds) in enumerate(zip(eval_fearures, clfs_preds, quant_scores)):
        example = ex_feature.example
        docId = example.docId
        subtok_pos_to_orig_tok_pos = ex_feature.subtok_pos_to_orig_tok_pos
        min_pos = min([start for (start, _) in subtok_pos_to_orig_tok_pos.keys()])
        good_example = True
        clfs_rows = {}
        for clf_id, clf in enumerate(clfs):
            clf_start = ex_clfs_preds[clf_id * 2]
            clf_end = ex_clfs_preds[clf_id * 2 + 1]
            if clf_start < min_pos or clf_end < min_pos:
                continue
            start = [pos for (start, end), pos in subtok_pos_to_orig_tok_pos.items() if start <= clf_start <= end]
            end = [pos for (start, end), pos in subtok_pos_to_orig_tok_pos.items() if start <= clf_end <= end]
            if start > end:
                continue
            if not start or not end:
                continue
            start = example.tokens[start[0]].start
            end = example.tokens[end[0]].end
            sent_start = example.sent.start
            text = example.sent.text[start - sent_start:end - sent_start]
            clfs_rows[clf] = (start, end, text)

        if 'Quantity' not in clfs_rows:
            continue
        start, end, text = clfs_rows['Quantity']
        quant_row = {
            'docId': docId, 'annotSet': annotSet, 'annotType': 'Quantity',
            'startOffset': start, 'endOffset': end, 'annotId': f'T1-{annotSet}', 'text': text
        }
        other = {}
        if 'Unit' in clfs_rows:
            other['unit'] = clfs_rows['Unit'][2]
        mods = [quant_mod for i, quant_mod in quant_id_to_labs.items() if ex_quant_preds[i] > 0.5]
        if ex_id < 10 and example.mods != 'O':
            print(example.mods)
            print({quant_mod: ex_quant_preds[i] for i, quant_mod in quant_id_to_labs.items()})
        if mods:
            other['mods'] = mods
        quant_row['other'] = json.dumps(other)

        predictions[docId].append(quant_row)

        if 'MeasuredProperty' in clfs_rows:
            start, end, text = clfs_rows['MeasuredProperty']
            other = {'HasQuantity': f'T1-{annotSet}'}
            mp_row = {
                'docId': docId, 'annotSet': annotSet, 'annotType': 'MeasuredProperty',
                'startOffset': start, 'endOffset': end, 'annotId': f'T2-{annotSet}', 'text': text,
                'other': json.dumps(other)
            }
            predictions[docId].append(mp_row)

        if 'MeasuredEntity' in clfs_rows:
            start, end, text = clfs_rows['MeasuredEntity']
            other = {'HasProperty': f'T2-{annotSet}'} if 'MeasuredProperty' in clfs_rows else {'HasQuantity': f'T1-{annotSet}'}
            me_row = {
                'docId': docId, 'annotSet': annotSet, 'annotType': 'MeasuredEntity',
                'startOffset': start, 'endOffset': end, 'annotId': f'T3-{annotSet}', 'text': text,
                'other': json.dumps(other)
            }
            predictions[docId].append(me_row)

        if 'Qualifier' in clfs_rows:
            start, end, text = clfs_rows['Qualifier']
            other = {'Qualifies': f'T2-{annotSet}'} if 'MeasuredProperty' in clfs_rows else {'Qualifies': f'T3-{annotSet}'} if 'MeasuredEntity' in clfs_rows else {'Qualifies': f'T1-{annotSet}'}
            qu_row = {
                'docId': docId, 'annotSet': annotSet, 'annotType': 'Qualifier',
                'startOffset': start, 'endOffset': end, 'annotId': f'T4-{annotSet}', 'text': text,
                'other': json.dumps(other)
            }
            predictions[docId].append(qu_row)

        for qualif_id, annotType in enumerate(['Quantity', 'MeasuredProperty', 'MeasuredEntity']):
            if f'{annotType}Qualifier' in clfs_rows:
                start, end, text = clfs_rows[f'{annotType}Qualifier']

                other = {'Qualifies': f'{T[annotType]}-{annotSet}'}
                qu_row = {
                    'docId': docId, 'annotSet': annotSet, 'annotType': 'Qualifier',
                    'startOffset': start, 'endOffset': end, 'annotId': f'T4{qualif_id}-{annotSet}', 'text': text,
                    'other': json.dumps(other)
                }
                predictions[docId].append(qu_row)

        annotSet += 1

    if os.path.exists(output_dir):
        os.system(f'rm -r {output_dir}/*')

    for docId, rows in predictions.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, docId + '.tsv'), sep='\t', index=False, quotechar='"')

    if compute_metrics:
        for key in metrics:
            metrics[key] /= nb_eval_steps
        try:
            os.system(f'python {eval_script_path} -i ./ -g ../MeasEval/data/trial/tsv/ -s {output_dir}/ > {output_dir}/scores.txt')

            with open(f'{output_dir}/scores.txt') as f:
                score = float([line.rstrip().split()[-1] for line in f.readlines()[-10:] if line.startswith('Overall Score F1 (Overlap):')][-1])
                metrics['score'] = score
        except:
            metrics['score'] = 0.0

        if cur_train_mean_loss is not None:
            metrics.update(cur_train_mean_loss)
    else:
        metrics = {}

    model.train()

    return metrics


def main(args):
    if not args.dont_split_qualifier:
        args.clfs = [
            "Quantity",
            "MeasuredEntity",
            "MeasuredProperty",
            "Unit",
            "QuantityQualifier",
            "MeasuredEntityQualifier",
            "MeasuredPropertyQualifier"
        ]
    else:
        args.clfs = [
            "Quantity",
            "MeasuredEntity",
            "MeasuredProperty",
            "Unit",
            "Qualifier"
            ]
    args.add_double_sentences = args.add_double_sentences.lower() == 'true'
    if args.do_train and os.path.exists(args.output_dir):
        from glob import glob
        model_weights = glob(os.path.join(args.output_dir, '*.bin'))
        if model_weights:
            print(f'{model_weights}: already computed: skipping ...')
            return
        else:
            print(f'already existing {args.output_dir} but without model weights: skipping ...')
            return

    device = torch.device("cuda" if not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps parameter should be >= 1"
        )

    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if args.do_train:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True."
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.do_train or args.do_validation:
        raise ValueError(args.output_dir, 'output_dir already exists')

    suffix = datetime.now().isoformat().replace('-', '_').replace(
        ':', '_').split('.')[0].replace('T', '-')
    if args.do_train:
        train_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'train'
            )
        )
        dev_writer = SummaryWriter(
            log_dir=os.path.join(
                args.output_dir, f'tensorboard-{suffix}', 'dev'
            )
        )

        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"train_{suffix}.log"), 'w')
        )
        eval_logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"scores_{suffix}.log"), 'w')
        )
    else:
        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"eval_{suffix}.log"), 'w')
        )

    logger.info(args)
    logger.info(json.dumps(vars(args), indent=4))
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))


    clfs = sorted(args.clfs)

    model_name = args.model_name
    data_processor = DataProcessor(os.path.join(args.data_dir, 'train/text/'), split_qualifier=not args.dont_split_qualifier, add_double_sentences=args.add_double_sentences)

    train_docs, train_annotations = os.path.join(args.data_dir, 'train/text'), os.path.join(args.data_dir, 'train/tsv')
    dev_docs, dev_annotations = os.path.join(args.data_dir, 'trial/txt'), os.path.join(args.data_dir, 'trial/tsv')
    test_docs, test_annotations = os.path.join(args.data_dir, 'eval/text'), os.path.join(args.data_dir, 'eval/single_sents_tsv')

    if args.do_train:
        config = configs[model_name]
        config = config.from_pretrained(
            model_name,
            hidden_dropout_prob=args.dropout
        )
        print(model_name)
        if args.ckpt_path != '':
            model_path = args.ckpt_path
        else:
            model_path = model_name
        model = models[model_name].from_pretrained(
            model_path, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            args=args,
            data_processor=data_processor,
            config=config
        )

        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    param for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)
                ],
                'weight_decay': float(args.weight_decay)
            },
            {
                'params': [
                    param for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=float(args.learning_rate),
            eps=1e-6,
            betas=(0.9, 0.98),
            correct_bias=True
        )

        train_features = model.convert_dataset_to_features(
            train_docs, train_annotations, logger, add_question=args.add_question
        )

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(
                train_features, key=lambda f: np.sum(f.input_mask)
            )
        else:
            random.shuffle(train_features)

        train_dataloader, clfs_labels_ids, quant_labels_ids = \
            get_dataloader_and_tensors(train_features, args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_batches) // args.gradient_accumulation_steps * \
                args.num_train_epochs

        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        if args.lr_scheduler == 'linear_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps
            )
        elif args.lr_scheduler == 'constant_warmup':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps
            )

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.do_validation:
            dev_features = model.convert_dataset_to_features(
                dev_docs, dev_annotations, logger, add_question=args.add_question, part='dev'
            )
            logger.info("***** Dev *****")
            logger.info("  Num examples = %d", len(dev_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            dev_dataloader, dev_clfs_labels_ids, dev_quant_labels_ids = \
                get_dataloader_and_tensors(dev_features, args.eval_batch_size)

            test_features = model.convert_dataset_to_features(
                test_docs, test_annotations, logger, add_question=args.add_question, part='test'
                )
            logger.info("***** Test *****")
            logger.info("  Num examples = %d", len(test_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            test_dataloader, test_clfs_labels_ids, test_quant_labels_ids = \
                get_dataloader_and_tensors(test_features, args.eval_batch_size)

        best_result = 0.0
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        start_time = time.time()
        global_step = 0

        model.to(device)
        lr = float(args.learning_rate)

        for epoch in range(1, 1 + args.num_train_epochs):
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            cur_train_loss = defaultdict(float)

            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, scheduler.get_lr()[0]))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)

            train_bar = tqdm(
                train_batches, total=len(train_batches),
                desc='training ... '
            )
            for step, batch in enumerate(
                train_bar
            ):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, token_type_ids, \
                clfs_labels, quant_labels = batch
                train_loss, _, _ = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=input_mask,
                    input_labels={'labels': clfs_labels, 'quant_labels': quant_labels},
                    weighting_mode=args.weighting_mode
                )
                loss = train_loss['total'].mean().item()
                for key in train_loss:
                    cur_train_loss[key] += train_loss[key].mean().item()
                train_bar.set_description(f'training... [epoch == {epoch} / {args.num_train_epochs}, loss == {loss}]')

                loss_to_optimize = train_loss['total']

                if args.gradient_accumulation_steps > 1:
                    loss_to_optimize = \
                        loss_to_optimize / args.gradient_accumulation_steps

                loss_to_optimize.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm
                )

                tr_loss += loss_to_optimize.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if args.do_validation and (step + 1) % eval_step == 0:
                    logger.info(
                        'Ep: {}, Stp: {}/{}, usd_t={:.2f}s, loss={:.6f}'.format(
                            epoch, step + 1, len(train_batches),
                            time.time() - start_time, tr_loss / nb_tr_steps
                        )
                    )

                    cur_train_mean_loss = {}
                    for key, value in cur_train_loss.items():
                        cur_train_mean_loss[f'train_{key}_loss'] = \
                            value / nb_tr_steps

                    dev_predictions = os.path.join(args.output_dir, 'dev_predictions')

                    metrics = predict(
                        model, dev_dataloader, dev_predictions,
                        dev_features, args,
                        cur_train_mean_loss=cur_train_mean_loss,
                        logger=eval_logger
                    )

                    metrics['global_step'] = global_step
                    metrics['epoch'] = epoch
                    metrics['learning_rate'] = scheduler.get_lr()[0]
                    metrics['batch_size'] = \
                        args.train_batch_size * args.gradient_accumulation_steps

                    for key, value in metrics.items():
                        dev_writer.add_scalar(key, value, global_step)

                    weights = torch.exp(model.clfs_weights) / torch.sum(torch.exp(model.clfs_weights))
                    for clf_id, clf in enumerate(clfs):
                        train_writer.add_scalar(f'{clf}Weight', weights[clf_id].item(), global_step)
                    train_writer.add_scalar(f'QuantModsWeight', weights[-1].item(), global_step)

                    for clf_id, clf in enumerate(clfs):
                        logger.info(f'{clf} weight: {weights[clf_id].item()}')
                    logger.info(f'quant_mods weight: {weights[-1].item()}')
                    logger.info(f"dev %s (lr=%s, epoch=%d): %.2f" %
                        (
                            'score',
                            str(scheduler.get_lr()[0]), epoch,
                            metrics['score'] * 100.0
                        )
                    )

                    if metrics['score'] > best_result:
                        logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f -> %.2f" %
                            (
                                'score',
                                str(scheduler.get_lr()[0]), epoch,
                                best_result * 100.0, metrics['score'] * 100.0
                            )
                        )
                        best_result = metrics['score']
                        # output_model_file = os.path.join(
                        #     args.output_dir,
                        #     WEIGHTS_NAME
                        # )
                        # save_model(args, model, output_model_file)
                        best_dev_predictions = os.path.join(args.output_dir, 'best_dev_predictions')
                        if os.path.exists(best_dev_predictions):
                            os.system(f'rm -r {best_dev_predictions}')
                        os.system(f'mv {dev_predictions} {best_dev_predictions}')
                        test_predictions = os.path.join(args.output_dir, 'best_test_predictions')
                        predict(
                            model, test_dataloader, test_predictions,
                            test_features, args, compute_metrics=False
                        )
                    dev_writer.add_scalar('best_score', best_result, global_step)

            if args.log_train_metrics:
                metrics = predict(
                    model, train_dataloader, os.path.join(args.output_dir, 'train_predictions'),
                    train_features, args,
                    logger=logger
                )
                metrics['global_step'] = global_step
                metrics['epoch'] = epoch
                metrics['learning_rate'] = scheduler.get_lr()[0]
                metrics['batch_size'] = \
                    args.train_batch_size * args.gradient_accumulation_steps

                for key, value in metrics.items():
                    train_writer.add_scalar(key, value, global_step)

    if args.do_eval:
        assert args.ckpt_path != ''
        test_docs, test_annotations = args.eval_input_txt_dir, args.eval_input_tsv_dir
        model = models[model_name].from_pretrained(
            args.ckpt_path, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            args=args,
            data_processor=data_processor
        )
        model.to(device)
        test_features = model.convert_dataset_to_features(
            test_docs, test_annotations, test_logger, add_question=args.add_question, part='test'
        )
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_dataloader, test_clfs_labels_ids, test_quant_labels_ids = \
            get_dataloader_and_tensors(test_features, args.eval_batch_size)

        metrics = predict(
            model, test_dataloader,
            os.path.join(args.output_dir, args.eval_output_dir),
            test_features, args,
            compute_metrics=False
        )



def save_model(args, model, output_model_file):
    start = time.time()
    model_to_save = \
        model.module if hasattr(model, 'module') else model

    output_config_file = os.path.join(
        args.output_dir, CONFIG_NAME
    )
    torch.save(
        model_to_save.state_dict(), output_model_file
    )
    model_to_save.config.to_json_file(
        output_config_file
    )
    print(f'model saved in {time.time() - start} seconds to {output_model_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action='store_true', help='Whether to run training')
    parser.add_argument("--do_validation", action='store_true',
                        help='Whether to validate model during training process')
    parser.add_argument("--do_eval", action='store_true', help='Whether to run evaluation')

    parser.add_argument("--model_name", default='roberta-large', type=str, required=False)
    parser.add_argument("--data_dir", default='../MeasEval/data/', type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_input_tsv_dir", default='', help='Directory containing .data files to predict')
    parser.add_argument("--eval_input_txt_dir", default='', help='Directory containing .data files to predict')
    parser.add_argument("--eval_output_dir", default='',
                        help='Subdirectory name where predictions will be saved')
    parser.add_argument("--ckpt_path", type=str, default='', help='Path to directory containig pytorch.bin checkpoint')

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--eval_per_epoch", default=4, type=int,
                        help="How many times to do validation on dev set per epoch")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.\n"
                             "Sequences longer than this will be truncated, and sequences shorter\n"
                             "than this will be padded.")

    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup.\n"
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="maximal gradient norm")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="weight_decay coefficient for regularization")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=2021,
                        help="random seed for initialization")
    parser.add_argument("--log_train_metrics", action="store_true",
                        help="compute metrics for train set too")

    parser.add_argument("--weighting_mode", default='softmax',
                        type=str, required=False, choices=['softmax', 'equal', 'rsqr+log'])

    parser.add_argument("--pool_type", default='max', type=str, choices=['max', 'mean', 'first'])
    parser.add_argument("--lr_scheduler", default='linear_warmup', type=str, choices=['linear_warmup', 'constant_warmup'])

    parser.add_argument("--concat_quantity_embeddings", default=False, type=bool)
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--num_train_epochs", default=50, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--quant_ncls", default=10, type=int)
    parser.add_argument("--max_seq_len", default=256, type=int)
    parser.add_argument("--clfs", default='', type=str)

    parser.add_argument("--dont_split_qualifier", action='store_true')
    parser.add_argument(
        "--quant_classes",
        default=[
             "HasTolerance", "IsApproximate", "IsCount", "IsList", "IsMean", "IsMeanHasTolerance",
             "IsMeanIsRange", "IsMedian", "IsRange", "IsRangeHasTolerance"
        ],
        type=list
    )
    parser.add_argument("--add_double_sentences", default='true', type=str, help="If true augmentation will be performed "
        "by inclusion of two sentences following each other")
    parser.add_argument("--add_question", default='Find measured entities and properties of marked quantity', type=str,
        help="Qustion to add before the sentence.")


    parsed_args = parser.parse_args()
    main(parsed_args)
