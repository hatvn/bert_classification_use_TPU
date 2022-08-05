import os.path
from dataclasses import dataclass
import jax.numpy as jnp
import optax
from typing import Any, Callable, Dict, Optional, Tuple
from flax import struct, traverse_util
from flax.training.common_utils import onehot
from transformers import AutoTokenizer
import datetime
from transformers import FlaxAutoModelForSequenceClassification, AutoConfig

from datasets import load_dataset
import jax
import jax.numpy as jnp
from flax.training import train_state


def read_sa_dataset(data_folder):
    dic_path = {
        'train': os.path.join(data_folder, 'train.csv'),
        'dev': os.path.join(data_folder, 'dev.csv'),
        'test': os.path.join(data_folder, 'test.csv')
    }
    dataset = load_dataset('csv', data_files=dic_path, column_names=["label", "text"])
    return dataset


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


def create_train_state(
    model: FlaxAutoModelForSequenceClassification,
    learning_rate_fn: Callable[[int], float],
    num_labels: int,
    weight_decay: float,
) -> train_state.TrainState:
    """Create initial training state."""
    class TrainState(train_state.TrainState):
        """Train state with an Optax optimizer.

        The two functions below differ depending on whether the task is classification
        or regression.

        Args:
          logits_fn: Applied to last layer to obtain the logits.
          loss_fn: Function to compute the loss.
        """
        logits_fn: Callable = struct.field(pytree_node=False)
        loss_fn: Callable = struct.field(pytree_node=False)

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
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

    return TrainState.create(
            apply_fn=model.__call__,
            params=model.params,
            tx=tx,
            logits_fn=lambda logits: logits.argmax(-1),
            loss_fn=cross_entropy_loss,
        )


def get_default_config():
    return {
        'data_folder': 'data/small_sen',
        'pretrained': 'bert-base-cased',
        'max_length': 64,
        'learning_rate': 1e-5,
        'epoch_num': 5,
        'batch_size': 64,
        'weight_decay': 0.01,
        'log_train_step': 50,
        'eval_step': 50
    }


def eval_step(batch, train_state):
    logits = train_state.apply_fn(**batch, params=train_state.params, train=False)[0]
    return logits


def train_step(batch, train_state, dropout_rng):
    labels = batch.pop('label')
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_func(params):
        logits = train_state.apply_fn(**batch, params=params, train=True, dropout_rng=dropout_rng)[0]
        return train_state.loss_fn(logits, labels)

    grad_fn = jax.value_and_grad(loss_func)
    loss, grad = grad_fn(train_state.params)
    new_state = train_state.apply_gradients(grads=grad)
    return new_state, new_dropout_rng, loss


def get_train_loader(ds, batch_size, rng_key):
    size = len(ds)
    perms = jax.random.permutation(rng_key, size)
    batch_num = size // batch_size
    for i in range(batch_num):
        start = i * batch_size
        end = i * batch_size + batch_size
        if end > size: # incomplete batch will be filled upto batch_size
            end = size
            start = size - batch_size
        batch = ds[perms[start: end]]
        batch_jax = {key: jnp.array(batch[key]) for key in batch}
        yield batch_jax


def get_eval_loader(ds, batch_size):
    size = len(ds)
    batch_num = size // batch_size
    for i in range(batch_num):
        start = i * batch_size
        end = i * batch_size + batch_size
        offset = 0
        if end > size:
            old_start = start
            start = size - batch_size
            end = size
            offset = old_start - start
        batch = ds[start: end]
        batch_jax = {key: jnp.array(batch[key]) for key in batch}
        yield batch_jax, offset


def get_accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    size = len(labels)
    count = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            count += 1
    return {'acc': count / size, 'total': size}


def train(config):
    """
    :param config: {'data_folder': }
    :return:
    """
    data_folder = config['data_folder']
    ds = read_sa_dataset(data_folder)
    print(ds)
    pretrained_path = config['pretrained']
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    def preprocess_record(examples):
        text = examples['text']
        return tokenizer(text, max_length=config['max_length'], padding="max_length", truncation=True)
    ds = ds.map(preprocess_record, remove_columns=['text'])
    train_ds = ds['train']
    dev_ds = ds['dev']

    m_config = AutoConfig.from_pretrained(
        pretrained_path, label_nums=2
    )
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        pretrained_path, config=m_config
    )
    learning_rate = config['learning_rate']
    epoch_num = config['epoch_num']
    batch_size = config['batch_size']
    warmup_steps = config.get('warmup_steps', 3)
    weight_decay = config.get('weight_decay', 0.9)

    rng_key = jax.random.PRNGKey(10)
    rng_key, dropout_key = jax.random.split(rng_key)
    learning_rate_func = create_learning_rate_fn(
        len(train_ds), batch_size, epoch_num, warmup_steps, learning_rate
    )
    train_state = create_train_state(model, learning_rate_fn=learning_rate_func, num_labels=2, weight_decay=weight_decay)
    rng_key, batch_key = jax.random.split(rng_key)
    jix_train_step = jax.jit(train_step, donate_argnums=(1,))
    jix_eval_step = jax.jit(eval_step)
    cur_step = 0
    t1 = datetime.datetime.now()
    for epo in range(epoch_num):
        print('--------------------start epo=%d---------------' % epo)
        rng_key, batch_key = jax.random.split(rng_key)
        train_loader = get_train_loader(train_ds, batch_size, batch_key)

        for l_step, batch in enumerate(train_loader):
            train_state, dropout_key, loss = jix_train_step(batch, train_state, dropout_key)
            cur_step += 1
            if cur_step % config['log_train_step'] == 0:
                print('epo: %d, step: %d, loss = %f' % (epo, cur_step, loss))
            if cur_step % config['eval_step'] == 0 and cur_step > 0:
                print('start evaluating data at epo: %d, step: %d' % (epo, cur_step))
                t_predictions = []
                t_labels = []
                t_loss = 0
                for batch, start_offset in get_eval_loader(dev_ds, batch_size):
                    labels = batch.pop('label')
                    logits = jix_eval_step(batch, train_state)
                    loss = optax.softmax_cross_entropy(logits, onehot(labels, num_classes=2))
                    t_loss += jnp.sum(loss)
                    predictions = jnp.argmax(logits, axis=-1)
                    labels = labels[start_offset:]
                    predictions = predictions[start_offset:]
                    t_labels.extend(labels.tolist())
                    t_predictions.extend(predictions.tolist())
                print('number of records in dev: ', len(t_predictions))
                metric = get_accuracy(t_labels, t_predictions)
                print('eval result: loss: %s, metric=%s' % (str(t_loss), str(metric)))
    t2 = datetime.datetime.now()
    print('exe time: %f seconds' % (t2 - t1).total_seconds())


def main():
    config = get_default_config()
    config['data_folder'] = 'data/small_sen'
    config['pretrained'] = 'distilbert-base-uncased'
    train(config)


if __name__ == '__main__':
    main()

