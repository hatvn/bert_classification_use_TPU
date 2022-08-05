import os
from datasets import load_dataset
from transformers import AutoConfig, Trainer
import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from torch.utils.data import DataLoader


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
          'pretrained': 'bert-base-cased',
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
      data_loader = DataLoader(ds, batch_size=3, collate_fn=DataCollatorWithPadding(tokenizer))
      count = 0
      for batch in data_loader:
        print('---------------------------------')
        print('length of this batch: ', batch['input_ids'].size(-1))
        for item in batch['input_ids']:
          leng = (item != 0).sum()
          print('leng = ', leng)
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


def train(config):
    sa_ds = read_sa_dataset(config['data_folder'])
    pretrained = 'distilbert-base-uncased'
    max_length = config['max_length']
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def preprocess_examples(examples):
        text = examples['text']
        result = tokenizer(text, max_length=max_length, padding=True, truncation=True)
        result['labels'] = examples['label']
        return result

    tok_sa_ds = sa_ds.map(preprocess_examples, remove_columns=['label', 'text'])
    train_ds = tok_sa_ds['train']
    dev_ds = tok_sa_ds['dev']
    test_ds = tok_sa_ds['test']

    print_random_records(train_ds, tokenizer)
    training_args = TrainingArguments(
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

    m_config = AutoConfig.from_pretrained(
            pretrained, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, config=m_config)

    trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_ds,  # training dataset
            eval_dataset=dev_ds,  # evaluation dataset
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        )
    t1 = datetime.datetime.now()
    trainer.train()
    t2 = datetime.datetime.now()
    print('training time: %f seconds' % (t2 - t1).total_seconds())


def main():
    config = get_default_config()
    config['data_folder'] = 'data/small_sen'
    config['pretrained'] = 'distilbert-base-uncased'
    train(config)


if __name__ == '__main__':
    main()
