import torch
# TPU-specific libraries (must-haves)
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from datasets import load_dataset
from transformers import AutoConfig, Trainer
import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import default_data_collator


def read_sa_dataset(data_folder):
  dic_path = {
      'train' : os.path.join(data_folder, 'train.csv'),
      'dev': os.path.join(data_folder, 'dev.csv'),
      'test': os.path.join(data_folder, 'test.csv')
  }
  dataset = load_dataset('csv', data_files=dic_path, column_names=['label', 'text'])
  return dataset


def get_default_config():
    return {
          'data_folder': 'data/small_sen',
          'pretrained': 'distilbert-base-uncased',
          'max_length': 64,
          'learning_rate': 1e-5,
          'epoch_num': 5,
          'batch_size': 16,
          'weight_decay': 0.01,
          'log_train_step': 100,
          'eval_step': 200
    }


def print_random_records(ds, tokenizer):
      print('padding_id: ', tokenizer.pad_token_id)
      data_loader = DataLoader(ds, batch_size=3, collate_fn=default_data_collator)
      count = 0
      for batch in data_loader:
        print('---------------------------------')
        print(batch)
        print('shape of this batch: ', batch['input_ids'].shape)
        for item in batch['input_ids']:
          leng = (item != 0).sum()
          print('leng = ', leng.item())
          print('text: ', tokenizer.decode(item.tolist()))
        count += 1
        if count == 3:
          break


def compute_metrics(pred):
      labels = pred.label_ids
      preds = pred.predictions.argmax(-1)
      size = len(labels)
      correct_count = (labels==preds).sum().item()
      return {'acc': correct_count / size}


def prepare_data(config):
    torch.set_default_tensor_type('torch.FloatTensor')
    sa_ds = read_sa_dataset(config['data_folder'])
    pretrained = config['pretrained']
    max_length = config['max_length']
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def preprocess_examples(examples):
        text = examples['text']
        result = tokenizer(text, max_length=max_length, padding='max_length', truncation=True)
        result['labels'] = examples['label']
        return result

    tok_sa_ds = sa_ds.map(preprocess_examples, remove_columns=['label', 'text'])
    train_ds = tok_sa_ds['train']
    dev_ds = tok_sa_ds['dev']
    test_ds = tok_sa_ds['test']

    print_random_records(train_ds, tokenizer)
    return train_ds, dev_ds, test_ds


def train_with_data(index, config, train_ds, dev_ds):
    pretrained = config['pretrained']
    training_args = TrainingArguments(
            dataloader_drop_last=True,
            output_dir='save_models/pt',  # output directory
            num_train_epochs=config['epoch_num'],  # total # of training epochs
            per_device_train_batch_size=config['batch_size'],  # batch size per device during training
            per_device_eval_batch_size=config['batch_size'],  # batch size for evaluation
            warmup_steps=config.get('warm_up', 3),  # number of warmup steps for learning rate scheduler
            weight_decay=config.get('weight_decay', 0.01),  # strength of weight decay
            logging_dir='save_models/pt',  # directory for storing logs
            learning_rate=config['learning_rate'],
            gradient_accumulation_steps=1,
            eval_steps=config['eval_step'],
            evaluation_strategy='steps'
        )

    device = xm.xla_device()
    m_config = AutoConfig.from_pretrained(
            pretrained, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, config=m_config)
    model.to(device)

    trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_ds,  # training dataset
            eval_dataset=dev_ds,  # evaluation dataset
            compute_metrics=compute_metrics
        )
    t1 = datetime.datetime.now()
    trainer.train()
    t2 = datetime.datetime.now()
    print('training time: %f seconds' % (t2 - t1).total_seconds())


def train(config):
    train_ds, dev_ds, test_ds = prepare_data(config)
    xmp.spawn(train_with_data, args=(config, train_ds, dev_ds,), nprocs=8, start_method='fork')


if __name__ == '__main__':
    config = get_default_config()
    config['data_folder'] = 'data/small_sen'
    train(config)