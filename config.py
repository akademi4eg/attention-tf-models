from easydict import EasyDict

config = EasyDict()
config.batch_size = 512
config.cell_units = 256
config.encoder_layers = 3
config.decoder_layers = 2
config.attention_units = 256
config.decoder_emb_size = 256
config.pool_step = 1
config.conv_units = 0
config.pass_state = True
config.max_iter = 50
config.learn_rate = 1e-6
config.clip_norm = 20
config.sampling_prob = 1.0
config.dropout_in = 0.0
config.dropout_out = 0.1
config.timit = 'path/to/TIMIT'