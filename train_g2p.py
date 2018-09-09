import logging
import tensorflow as tf

from config import config
from model import E2EModel
from cmu_source import CMUDataSource
from units import ABC

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('train')
    data_source = CMUDataSource(ABC(), ABC())
    if not data_source.load():
        data_source.compile_abc()
        data_source.save()
    config.features_num = len(data_source.abc_gr.vocab)
    with tf.Session() as sess:
        model = E2EModel(config, data_source.abc_ph, sess)
        sess.run(tf.global_variables_initializer())
        model.load()
        logger.info('Starting training.')
        best_loss, train_cer, infer_cer, wer = model.get_metrics(data_source.batches(config.batch_size, False))
        logger.info(f'Initial loss is {best_loss:2.2f}, '
                    f'train CER {100 * train_cer:2.0f}%, infer CER {100 * infer_cer:2.0f}%, '
                    f'WER {100 * wer:2.0f}%')
        for ep in range(1, 1001):
            model.train(data_source.batches(config.batch_size, True))
            loss, train_cer, infer_cer, wer = model.get_metrics(data_source.batches(config.batch_size, False))
            wer = 1 - wer
            logger.info(f'After epoch {ep} loss is {loss:2.2f} (best: {best_loss:2.2f}), '
                        f'train CER {100 * train_cer:2.0f}%, infer CER {100 * infer_cer:2.0f}%, '
                        f'WER {100 * wer:2.0f}%')
            if loss < best_loss:
                best_loss = loss
                model.save()
                logger.info('Saving best model')
