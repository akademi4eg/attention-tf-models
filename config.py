from easydict import EasyDict

config = EasyDict()
config.batch_size = 128
config.cell_units = 512
config.encoder_layers = 3
config.decoder_layers = 2
config.attention_units = 512
config.decoder_emb_size = 512
config.pool_step = 1
config.conv_units = 0
config.pass_state = True
config.max_iter = 50
config.learn_rate = 1e-3
config.clip_norm = 20
