import datetime

import torch
import os
import numpy as np
# TPU-specific libraries (must-haves)
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu


from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_default_flag():
    return {
        'seed': 123673,
        'save_folder': 'models/tpu_wo_trainer',
        'batch_size': 20,
        'data_folder': 'data/small_sen',
        'pretrained': 'distilbert-base-uncased',
        'max_length': 64,
        'lr': 1e-5,
        'epoch_num': 5,
        'weight_decay': 0.01,
        'log_train_step': 100,
        'eval_step': 20
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


def add_dataset(ds, add_size):
    new_item = ds[-1].copy()
    new_item['labels'] = -100
    result = ds
    for i in range(add_size):
        result = result.add_item(new_item)
    return result


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

    if xm.is_master_ordinal():
        print_random_records(train_ds, tokenizer)
    return train_ds, dev_ds, test_ds


def read_sa_dataset(data_folder):
    dic_path = {
      'train' : os.path.join(data_folder, 'train.csv'),
      'dev': os.path.join(data_folder, 'dev.csv'),
      'test': os.path.join(data_folder, 'test.csv')
    }
    return load_dataset('csv', data_files=dic_path, column_names=['label', 'text'])


def evaluate_model(model, mp_dev_loader):
    model.eval()
    acc = 0
    total_count = 0
    total_loss = 0
    for i, batch in enumerate(mp_dev_loader):
        labels = batch['labels']
        with torch.no_grad():
            output = model(**batch)
        logits = output.logits # B x 2
        max_preds = torch.argmax(logits, dim=-1)
        correct_count = (labels == max_preds).sum().item()
        acc += correct_count
        ignore_count = (labels == -100).sum().item()
        total_count += labels.shape[0] - ignore_count
        total_loss += output.loss.item()
        #if total_step < 3:
        #    input_ids = batch['input_ids']
        #    print('rank: ', xm.get_ordinal(), 'input: ', input_ids[0][:10])
    total_acc = xm.mesh_reduce('correct', acc, np.sum)
    test_size = xm.mesh_reduce('total', total_count, np.sum)
    total_loss = xm.mesh_reduce('total_lost', total_loss, np.sum)
    accuracy = total_acc / test_size
    return {'loss': total_loss, 'correct': total_acc, 'total': test_size, 'accuracy': accuracy}


def train(index, flag):
    # this is set to ensure that all the random numbers in each process are the same
    torch.manual_seed(flag['seed'])
    train_ds, dev_ds, test_ds = prepare_data(flag)
    xm.master_print(f'train_size: {len(train_ds)}, dev_size: {len(dev_ds)}')
    real_batch_size = flag['batch_size'] * xm.xrt_world_size()

    # we add more data to make sure that size % real_batch_size == 0
    # added examples having labels=-100, which is ignored in computing loss
    add_train_size = real_batch_size - len(train_ds) % real_batch_size
    if add_train_size > 0:
        train_ds = add_dataset(train_ds, add_train_size)
    add_dev_size = real_batch_size - len(dev_ds) % real_batch_size
    if add_dev_size > 0:
        dev_ds = add_dataset(dev_ds, add_dev_size)
    xm.master_print(f'train_size: {len(train_ds)}, dev_size: {len(dev_ds)}')

    dev_sampler = DistributedSampler(dev_ds, num_replicas=xm.xrt_world_size(),
                                     rank=xm.get_ordinal(), shuffle=False)
    # NOTE THAT as we train over 8 TPUS, the real batch_size = batch_size * 8
    train_sampler=DistributedSampler(train_ds, num_replicas=xm.xrt_world_size(),
                                     rank=xm.get_ordinal(), shuffle=True)

    train_loader = DataLoader(train_ds, batch_size=flag['batch_size'], sampler=train_sampler,
                              drop_last=True, collate_fn=default_data_collator)
    dev_loader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=flag['batch_size'], drop_last=True,
                            shuffle=False, collate_fn=default_data_collator)

    pretrained = flag['pretrained']
    m_config = AutoConfig.from_pretrained(
        pretrained, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, config=m_config)
    device = xm.xla_device()
    model = model.to(device)
    lr = flag['lr'] * xm.xrt_world_size()  # lr is multiplied by 8 times

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": flag['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer.zero_grad()

    gradient_accumulation_steps = flag.get('gradient_accumulation_steps', 1)
    train_size = len(train_ds)

    t_total = flag['epoch_num'] * (train_size / real_batch_size / gradient_accumulation_steps)
    warmup_steps = flag.get('warmup_steps', 0)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    xm.master_print('start training now ...')
    glb_step = 0

    mp_device_loader = pl.MpDeviceLoader(train_loader, device)
    mp_dev_loader = pl.MpDeviceLoader(dev_loader, device)
    save_folder = flag['save_folder']
    best_metric = {'accuracy': -1}
    for epo in range(flag['epoch_num']):
        xm.master_print('------------start training at epo = %d---------------' % epo)
        total_step_per_epoch = 0
        total_size_per_epoch = 0
        for l_step, batch in enumerate(mp_device_loader):
            total_step_per_epoch += 1
            total_size_per_epoch = batch['input_ids'].shape[0]
            model.train()
            output = model(**batch)
            loss = output.loss
            if gradient_accumulation_steps > 1:
                loss = loss/gradient_accumulation_steps
            loss.backward()
            if (glb_step + 1) % gradient_accumulation_steps == 0:
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()

            if glb_step % flag['eval_step'] == 0:
                instant_lr = scheduler.get_last_lr()[0]
                xm.master_print('----start eval---at step %d, lr= %f---' % (glb_step, instant_lr))
                metric = evaluate_model(model, mp_dev_loader)
                if metric['accuracy'] > best_metric['accuracy']:
                    best_metric = metric
                    xm.master_print('evaluation result, metric = ', metric)
                    xm.master_print('save model with new better accuracy: ', metric)
                    xm.rendezvous('save_model')
                    t1 = datetime.datetime.now()
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    model.save_pretrained(save_folder, save_function=xm.save)
                    t2 = datetime.datetime.now()
                    xm.master_print('time for saving: ', (t2 - t1).total_seconds())

            glb_step += 1
        if xm.is_master_ordinal():
            print('finished epo: %d' % epo)
        xm.master_print(f'total_step_per_epoch: {total_step_per_epoch}; total_size_per_epoch: {total_size_per_epoch}')


def main():
    flag = get_default_flag()
    xmp.spawn(train, (flag, ), nprocs=8, start_method='fork')


if __name__ == '__main__':
    main()
