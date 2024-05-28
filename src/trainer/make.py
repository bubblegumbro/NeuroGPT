#!/usr/bin/env python3

import os
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from transformers import TrainingArguments, TrainerCallback
from trainer.base import Trainer
from torch import nn

'''def compute_metric(eval_preds):
    preds, targets = eval_preds
    preds= np.where(preds != -100, preds, tokenizer.pad_token_id)
    targets= np.where(targets != -100, targets, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    correct = 0
    assert len(preds) == len(targets)
    for idx, pred in enumerate(preds):
        reference = targets[idx]
        reference = extract_ans(reference)
        extract_pred = extract_ans(pred)
        best_option = extract_pred
        if reference == best_option and reference != False:
            correct +=1 
    return {'accuracy': 1.0*correct/len(targets)}'''


'''def preprocess_logits_for_metrics(logits,labs):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        print(logits['outputs'].shape)
        if labs: 
            print('labs',labs.shape)
        pred_ids = torch.argmax(logits['outputs'], dim=-1)
        return pred_ids '''


import torch

def fixed_pca(data, n_components=100):
    """
    Perform PCA with a fixed number of components to ensure consistent output dimensions.
    """
    if data.dim() < 3:
        raise ValueError("Input data must have at least three dimensions: [batch, sequence_length, features]")

    # Flatten the data from [batch, sequence_length, features] to [batch*sequence_length, features]
    batch_size, seq_length, features = data.shape
    flat_data = data.reshape(-1, features)  # This combines batch and sequence_length

    # Compute SVD
    U, S, V = torch.pca_lowrank(flat_data, q=n_components)

    # Project the data onto the top 'n_components' principal components
    reduced_data = torch.matmul(flat_data, V[:, :n_components])

    # Ensure the number of components does not exceed the number of features
    n_components = min(n_components, V.shape[1])

    return reduced_data.reshape(batch_size, seq_length, n_components)

def preprocess_logits_for_metrics(logits, n_components=100):
    """
    Reduces dimensionality of logits to manage computational resources better while keeping the output size consistent.
    """
    if 'outputs' not in logits:
        raise KeyError("Expected key 'outputs' in logits dictionary")

    print("Original logits shape:", logits['outputs'].shape)
    
    reduced_logits = fixed_pca(logits['outputs'], n_components=n_components)
    print("Reduced logits shape:", reduced_logits.shape)

    return reduced_logits



    
class CSVLogCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.train_log_filepath = None
        self.eval_log_filepath = None

    def on_log(
        self,
        args,
        state,
        control,
        model,
        **kwargs
        ) -> None:

        if args.local_rank not in {-1, 0}:
            return

        if self.train_log_filepath is None:
            self.train_log_filepath = os.path.join(
                args.output_dir,
                'train_history.csv'
            )

            with open(self.train_log_filepath, 'a') as f:
                f.write('step,loss,lr\n')

        if self.eval_log_filepath is None:
            self.eval_log_filepath = os.path.join(
                args.output_dir,
                'eval_history.csv'
            )

            with open(self.eval_log_filepath, 'a') as f:
                f.write('step,loss,accuracy\n')

        is_eval = any('eval' in k for k in state.log_history[-1].keys())

        if is_eval:
            with open(self.eval_log_filepath, 'a') as f:
                f.write('{},{},{}\n'.format(
                        state.global_step,
                        state.log_history[-1]['eval_loss'],
                        state.log_history[-1]['eval_accuracy'] if 'eval_accuracy' in state.log_history[-1] else np.nan
                    )
                )

        else:

            with open(self.train_log_filepath, 'a') as f:
                f.write('{},{},{}\n'.format(
                        state.global_step,
                        state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else state.log_history[-1]['train_loss'],
                        state.log_history[-1]['learning_rate'] if 'learning_rate' in state.log_history[-1] else None
                    )
                )


def _cat_data_collator(features: List) -> Dict[str, torch.tensor]:

    if not isinstance(features[0], dict):
        features = [vars(f) for f in features] 

    return {
        k: torch.cat(
            [
                f[k]
                for f in features
            ]
        )
        for k in features[0].keys()
        if not k.startswith('__')
    }


def decoding_accuracy_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": round(accuracy, 3)
    }


def make_trainer(
    model_init,
    training_style,
    train_dataset,
    validation_dataset,
    do_train: bool = True,
    do_eval: bool = True,
    run_name: str = None,
    output_dir: str = None,
    overwrite_output_dir: bool = True,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    optim: str='adamw_hf',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    adam_beta1: float=0.9,
    adam_beta2: float=0.999,
    adam_epsilon: float=1e-8,
    max_grad_norm: float=1.0,
    per_device_train_batch_size: int = 8,  # Reduced batch size
    per_device_eval_batch_size: int = 8,   # Reduced batch size
    dataloader_num_workers: int = 4,
    max_steps: int = 400000,
    num_train_epochs: int = 1,
    lr_scheduler_type: str = 'linear',
    warmup_ratio: float = 0.01,
    #evaluation_strategy: str = 'steps',
    evaluation_strategy: str = 'no',
    prediction_loss_only: bool = 'False',
    loss_only: bool = 'True',
    logging_strategy: str = 'steps',
    save_strategy: str = 'steps',
    save_total_limit: int = 5,
    save_steps: int = 10000,
    logging_steps: int = 10000,
    eval_steps: int = None,
    logging_first_step: bool = True,
    greater_is_better: bool = True,
    seed: int = 1,
    fp16: bool = True,
    deepspeed: str = None,
    compute_metrics = None,
    gradient_accumulation_steps: int = 4,  # Added gradient accumulation
    **kwargs
    ) -> Trainer:
    """
    Make a Trainer object for training a model.
    Returns an instance of transformers.Trainer.
    
    See the HuggingFace transformers documentation for more details
    on input arguments:
    https://huggingface.co/transformers/main_classes/trainer.html

    Custom arguments:
    ---
    model_init: callable
        A callable that does not require any arguments and 
        returns model that is to be trained (see scripts.train.model_init)
    training_style: str
        The training style (ie., framework) to use.
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    train_dataset: src.batcher.dataset
        The training dataset, as generated by src.batcher.dataset
    validation_dataset: src.batcher.dataset
        The validation dataset, as generated by src.batcher.dataset

    Returns
    ----
    trainer: transformers.Trainer
    """
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        do_train=do_train,
        do_eval=do_eval,
        overwrite_output_dir=overwrite_output_dir,
        prediction_loss_only=prediction_loss_only,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        optim=optim,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        greater_is_better=greater_is_better,
        save_steps=save_steps,
        logging_strategy=logging_strategy,
        logging_first_step=logging_first_step,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps is not None else logging_steps,
        seed=seed,
        fp16=True,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Added gradient accumulation
        eval_accumulation_steps=4,
        deepspeed=deepspeed,
        **kwargs
    )

    data_collator = _cat_data_collator
    is_deepspeed = deepspeed is not None
    # TODO: custom compute_metrics so far not working in multi-gpu setting
    compute_metrics = decoding_accuracy_metrics if training_style=='decoding' and compute_metrics is None else compute_metrics

    trainer = Trainer(
        args=trainer_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=optimizers,
        is_deepspeed=is_deepspeed
    )

    trainer.add_callback(CSVLogCallback)

    return trainer
