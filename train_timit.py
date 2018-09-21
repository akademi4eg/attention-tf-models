import logging
import tensorflow as tf

from config import config
from model import E2EModel
from timit_source import TimitSource
from units import ABC
from features import raw_waveform

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('train')
    data_source = TimitSource('ti-chars', config, ABC(), raw_waveform)
    config.features_num = 1  # 1 channel waveform
    with tf.Session() as sess:
        model = E2EModel(config, data_source.abc, sess)
        sess.run(tf.global_variables_initializer())
        model.load()
        logger.info('Starting training.')
        best_loss, infer_cer, wer = model.get_metrics(data_source.batches(config.batch_size, False))
        wer = 1 - wer
        logger.info(f'Initial loss is {best_loss:2.3f}, '
                    f'CER {100 * infer_cer:2.2f}%, WER {100 * wer:2.2f}%')
        for ep in range(1, 1001):
            model.train(data_source.batches(config.batch_size, True))
            loss, infer_cer, wer = model.get_metrics(data_source.batches(config.batch_size, False))
            wer = 1 - wer
            logger.info(f'After epoch {ep} loss is {loss:2.3f} (best: {best_loss:2.3f}), '
                        f'CER {100 * infer_cer:2.2f}%, WER {100 * wer:2.2f}%')
            if loss < best_loss:
                best_loss = loss
                model.save()
                logger.info('Saving best model')
