import os.path
from dataclasses import dataclass

import flax.jax_utils
from itertools import chain
import jax.numpy as jnp
from typing import Any, Callable, Dict, Optional, Tuple
import optax
from typing import Any, Callable, Dict, Optional, Tuple
from flax import struct, traverse_util
from flax.training.common_utils import onehot
from transformers import AutoTokenizer
import datetime
from transformers import FlaxAutoModelForSequenceClassification, AutoConfig, FlaxBertForSequenceClassification

from datasets import load_dataset
import jax
import jax.numpy as jnp
from flax.training import train_state
#import evaluate Cannot import evaluate --> raise errors

Array = Any
PRNGKey = Any


def read_sa_dataset(data_folder):
    dic_path = {
        'train': os.path.join(data_folder, 'train.csv'),
        'dev': os.path.join(data_folder, 'dev.csv'),
        'test': os.path.join(data_folder, 'test.csv')
    }
    dataset = load_dataset('csv', data_files=dic_path, column_names=["label", "text"])
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
        'log_train_step': 50,
        'seed': 10,
        'eval_step': 20,
        'save_folder': 'models/flax_multi'
    }


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def create_train_state(model: FlaxAutoModelForSequenceClassification,
                       learning_rate_fn: Callable[[int], float],
                       num_labels: int,
                       weight_decay: float):

    class NewTrainState(train_state.TrainState):
        # note that this is mandatory if not pmap/jit will raise an error
        # the fact is that arguments for functions in jit/pmap must be pytree
        loss_func: Callable = struct.field(pytree_node=False)

    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    tx = optax.adamw(
        learning_rate=learning_rate_fn, b1=0.9, b2=0.999, eps=1e-6, weight_decay=weight_decay, mask=decay_mask_fn
    )

    def cross_entropy_loss(logits, labels):
        xentropy = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=num_labels))
        return jnp.mean(xentropy)

    return NewTrainState.create(
        apply_fn=model.__call__, params=model.params, tx=tx, loss_func=cross_entropy_loss
    )


def eval_step(state, batch):
    labels = batch.pop('label') # B
    output = state.apply_fn(**batch, params=state.params, train=False)
    loss = state.loss_func(output.logits, labels)
    preds = jnp.argmax(output.logits, axis=1)
    return loss, preds, labels


def train_step(state, batch, dropout_rng):
    labels = batch.pop('label')
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_func(params):
        logits = state.apply_fn(**batch,
            params=params, train=True, dropout_rng=dropout_rng
        ).logits
        loss = state.loss_func(logits, labels)
        return loss

    grad_fn = jax.value_and_grad(loss_func)
    loss, grad = grad_fn(state.params)  # compute loss and grad
    grad = jax.lax.pmean(grad, axis_name='batch')  # gather gradients from all cores to average
    new_state = state.apply_gradients(grads=grad)  # update state with gradient

    metrics = {'loss': jax.lax.psum(loss, axis_name='batch')}
    return new_state, new_dropout_rng, metrics


def get_train_loader(rand_key, ds, batch_size):
    local_device_count = jax.local_device_count()
    total_batch_size = batch_size * local_device_count
    total_size = len(ds)
    perms = jax.random.permutation(rand_key, total_size)
    num_steps = total_size // total_batch_size
    for step in range(num_steps + 1):
        start = step * total_batch_size
        end = step * total_batch_size + total_batch_size
        if end > total_size:  # if this is the incomplete batch --> offset with preceding examples to form a complete batch
            end = total_size
            start = total_size - total_batch_size
        indices = perms[start: end]
        batch = ds[indices]  # shape = (N,) + batch.shape[1:]
        # the line below is used to convert the first dimension=local_device_count to feed to each tpu
        batch = {key: jnp.array(batch[key]) for key in batch}  # convert list to jax numpy
        batch = jax.tree_util.tree_map(lambda x: x.reshape((local_device_count, batch_size) + x.shape[1:]), batch)
        yield batch


def get_eval_loader(ds, batch_size):
    local_device_count = jax.local_device_count()
    total_batch_size = batch_size * local_device_count
    total_size = len(ds)
    num_steps = total_size // total_batch_size
    for step in range(num_steps):
        start = step * total_batch_size
        end = start + total_batch_size
        batch = ds[start: end]
        batch = {key: jnp.array(batch[key]) for key in batch}
        batch = jax.tree_util.tree_map(lambda x: x.reshape((local_device_count, batch_size) + x.shape[1:]), batch)
        yield batch


def train(config):
    print('config to train: ', config)
    local_device_count = jax.local_device_count()
    print('number of local devices: ', local_device_count)
    data_folder = config['data_folder']
    sa_ds = read_sa_dataset(data_folder)
    pretrained = config['pretrained']
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def process_examples(examples):
        text = examples['text']
        input_dic = tokenizer(text, max_length=config['max_length'], padding='max_length', truncation=True)
        return input_dic

    sa_ds = sa_ds.map(process_examples, remove_columns=['text'])
    train_ds = sa_ds['train']
    dev_ds = sa_ds['dev']
    print('train_size: %d, dev_size: %d' % (len(train_ds), len(dev_ds)))
    num_labels = 2

    model_config = AutoConfig.from_pretrained(pretrained, label_nums=num_labels)
    model = FlaxAutoModelForSequenceClassification.from_pretrained(pretrained, config=model_config)

    learning_rate = config['learning_rate']
    epoch_num = config['epoch_num']
    batch_size = config['batch_size']
    warmup_steps = config.get('warmup_steps', 3)
    weight_decay = config.get('weight_decay', 0.9)

    learning_rate_func = create_learning_rate_fn(
        len(train_ds), batch_size, epoch_num, warmup_steps, learning_rate
    )
    rng_key = jax.random.PRNGKey(config['seed'])

    state = create_train_state(model, learning_rate_func, num_labels, weight_decay)
    # parameters in function in jax.pmap must be pytree: in_axes=(None, 0, None)
    # the donate_argnums is used when the argument has the same shape like the output and not used anymore
    # this is commonly used for pattern: params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params, state)
    p_train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step)

    rng_key, input_rng_key = jax.random.split(rng_key)
    rng_key, dropout_rng = jax.random.split(rng_key)
    glb_step = 0

    # replicate things to run pmap: state & dropout_rng
    # one thing worth remembering is that pmap: input = device_count x Batch x input
    # and output of the pmap is: device_count x output
    # in case we want to communicate between devices using: pmean, psum with axis string
    state = flax.jax_utils.replicate(state)
    dropout_rng = jax.random.split(dropout_rng, jax.local_device_count())

    best_accuracy = -1
    for epo in range(config['epoch_num']):
        print('------------start training at epo: %d-----------------' % epo)
        input_rng_key, epo_key = jax.random.split(input_rng_key)
        train_loader = get_train_loader(epo_key, train_ds, batch_size)
        for i, batch in enumerate(train_loader):
            glb_step += 1
            state, dropout_rng, metrics = p_train_step(state, batch, dropout_rng)
            if glb_step % config['eval_step'] == 1:
                metrics = flax.jax_utils.unreplicate(metrics)
                print('train metrics at step %d: %s' % (glb_step, str(metrics)))
                # evaluate
                print('start evaluating dev_size: ', len(dev_ds))
                eval_loader = get_eval_loader(dev_ds, batch_size)
                total_loss = 0
                total_correct = 0
                total_size = 0
                for _, batch in enumerate(eval_loader):
                    loss, preds, labels = p_eval_step(state, batch)

                    pred_flatten = preds.reshape(-1)
                    labels_flatten = labels.reshape(-1)
                    total_correct += (pred_flatten == labels_flatten).sum().item()
                    total_size += labels_flatten.shape[0]
                    total_loss += jnp.sum(loss).item()  # loss: device * loss
                eval_batch_size = jax.local_device_count() * batch_size
                num_leftover_samples = len(dev_ds) % eval_batch_size
                # leftover should be run in only 1 process
                if num_leftover_samples > 0 and jax.process_index() == 0:
                    batch = dev_ds[-num_leftover_samples:]
                    batch = {k: jnp.array(v) for k, v in batch.items()}
                    loss, preds, labels = eval_step(flax.jax_utils.unreplicate(state), batch)

                    pred_flatten = preds.reshape(-1)
                    labels_flatten = labels.reshape(-1)
                    total_correct += (pred_flatten == labels_flatten).sum().item()
                    total_size += labels_flatten.shape[0]
                    total_loss += jnp.sum(loss).item()
                accuracy = total_correct / total_size
                metric = {'loss': total_loss, 'total_size': total_size, 'acc': total_correct / total_size}
                print('Eval result: ', metric)
                if accuracy > best_accuracy and jax.process_index() == 0:
                    best_accuracy = accuracy
                    print('new metric found, save model')
                    params = jax.device_get(flax.jax_utils.unreplicate(state.params))
                    save_folder = config['save_folder']
                    model.save_pretrained(save_folder, params=params)


def main():
    config = get_default_config()
    config['data_folder'] = 'data/small_sen'
    config['pretrained'] = 'distilbert-base-uncased'
    train(config)


if __name__ == '__main__':
    main()