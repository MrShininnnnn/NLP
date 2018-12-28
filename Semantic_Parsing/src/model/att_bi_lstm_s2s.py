#! /usr/bin/python3.5
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@nyu.edu'

import tensorflow as tf

class AttBiLSTM(object):

	def __init__(self, config):
		
		self.config = config

		with tf.name_scope('input_layer'):
			self.x_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs') # batch size, encoder sequence length
			self.y_inputs = tf.placeholder(tf.int32, [None, None], name='decoder_inputs') # batch size, decoder sequence length
			self.y_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets') # batch size, decoder sequence length
			self.en_lengths = tf.placeholder(tf.int32, [None], name='encoder_sequence_lengths')
			self.de_lengths = tf.placeholder(tf.int32, [None], name='decoder_sequence_lengths')
			self.en_drop_rate = tf.placeholder_with_default(1.0, shape=(), name='encoder_drop_out')
			self.de_drop_rate = tf.placeholder_with_default(1.0, shape=(), name='decoder_drop_out')
			self.batch_size = tf.shape(self.x_inputs)[0]
			self.start_slice = tf.fill([self.batch_size], self.config.start_idx)
			self.de_max_len = tf.shape(self.y_inputs)[1]
			self.global_step = tf.Variable(0, trainable=False, name='global_step')

		with tf.name_scope('embedding_layer'):
			self.en_embedding = tf.Variable(
				initial_value=tf.random_uniform( 
					[self.config.all_vocab_size, self.config.embedding_size], 
					minval=-self.config.init_scale, 
					maxval=self.config.init_scale), 
				dtype=tf.float32, 
				name='en_embedding_matrix')
			self.em_en_inputs = tf.nn.embedding_lookup(self.en_embedding, self.x_inputs)
			self.de_embedding = tf.Variable(
				initial_value=tf.random_uniform( 
					[self.config.fl_vocab_size, self.config.embedding_size], 
					minval=-self.config.init_scale, 
					maxval=self.config.init_scale), 
				dtype=tf.float32, 
				name='de_embedding_matrix')
			self.em_de_inputs = tf.nn.embedding_lookup(self.de_embedding, self.y_inputs)

		with tf.name_scope('encoding_layer'):
			self.encoder_inputs = tf.transpose(self.em_en_inputs, [1, 0, 2], name='encoder_inputs')
			self.encoder_fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.en_num_units/2)
			self.encoder_bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.en_num_units/2)
			self.encoder_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
				self.encoder_fw_cell, 
				input_keep_prob=self.en_drop_rate, 
				output_keep_prob=self.en_drop_rate, 
				state_keep_prob=self.en_drop_rate)
			self.encoder_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
				self.encoder_bw_cell, 
				input_keep_prob=self.en_drop_rate, 
				output_keep_prob=self.en_drop_rate, 
				state_keep_prob=self.en_drop_rate)
			self.encoder_outputs, self.encoder_state = tf.nn.bidirectional_dynamic_rnn(
				cell_fw=self.encoder_fw_cell,
				cell_bw=self.encoder_bw_cell,
				inputs=self.encoder_inputs,
				sequence_length=self.en_lengths,
				dtype=tf.float32,
				time_major=True,
				scope='bi_lstm_encoder')
			self.encoder_outputs = tf.concat(self.encoder_outputs, -1)
			self.encoder_final_state_c = tf.concat([t[0] for t in self.encoder_state], axis=-1)
			self.encoder_final_state_h = tf.concat([t[1] for t in self.encoder_state], axis=-1)
			self.encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(
				c=self.encoder_final_state_c, h=self.encoder_final_state_h)
			self.decoder_initial_state = self.encoder_final_state
			self.tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
				self.encoder_outputs, 
				multiplier=self.config.beam_width, 
				name='tiled_encoder_outputs')
			self.tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
				self.encoder_final_state, 
				multiplier=self.config.beam_width, 
				name='tiled_encoder_final_state')
			self.tiled_en_lengths = tf.contrib.seq2seq.tile_batch(
				self.en_lengths, 
				multiplier=self.config.beam_width, 
				name='tiled_en_lengths')
			self.tiled_decoder_initial_state = self.tiled_encoder_final_state

		with tf.name_scope('decoding_layer'): 
			self.decoder_cell = tf.nn.rnn_cell.LSTMCell(self.config.de_num_units)
			self.decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
				self.decoder_cell, 
				input_keep_prob=self.de_drop_rate, 
				output_keep_prob=self.de_drop_rate, 
				state_keep_prob=self.de_drop_rate)
			self.decoder_dense = tf.layers.Dense(
				self.config.fl_vocab_size, 
				activation=self.config.dense_activation, 
				use_bias=self.config.dense_bias, 
				name='decoder_projection')

			with tf.name_scope('attention_layer'):
				self.att_inputs = tf.transpose(self.encoder_outputs, [1, 0, 2], name='attention_inputs')
				self.tiled_att_inputs = tf.contrib.seq2seq.tile_batch(
					self.att_inputs, 
					multiplier=self.config.beam_width, 
					name='tiled_att_inputs')

				with tf.variable_scope('shared_attention_mechanism'):
					self.att_mechanism = tf.contrib.seq2seq.BahdanauAttention(
						num_units=self.config.attention_num_units,
						memory=self.att_inputs,
						memory_sequence_length= self.en_lengths,
						normalize=self.config.AttentionNormalize,
						dtype=tf.float32, 
						name='attention_mechanism')

				with tf.variable_scope('shared_attention_mechanism', reuse=True):
					self.tiled_att_mechanism = tf.contrib.seq2seq.BahdanauAttention(
						num_units=self.config.attention_num_units,
								memory=self.tiled_att_inputs,
								memory_sequence_length=self.tiled_en_lengths,
								normalize=self.config.AttentionNormalize, 
								dtype=tf.float32, 
								name='tiled_attention_mechanism')

				with tf.variable_scope('shared_attention_cell'):
					self.att_cell = tf.contrib.seq2seq.AttentionWrapper(
						cell=self.decoder_cell,
						attention_mechanism=self.att_mechanism,
						attention_layer_size=self.config.attention_layer_size,
						name='attention_cell')
					self.att_decoder_initial_state = self.att_cell.zero_state(
						batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=self.encoder_final_state)

				with tf.variable_scope('shared_attention_cell', reuse=True):
					self.tiled_att_cell = tf.contrib.seq2seq.AttentionWrapper(
						cell=self.decoder_cell,
						attention_mechanism=self.tiled_att_mechanism,
						attention_layer_size=self.config.attention_layer_size,
						name='attention_cell')
					self.tiled_att_decoder_initial_state = self.tiled_att_cell.zero_state(
						batch_size=self.batch_size * self.config.beam_width, dtype=tf.float32)
					self.tiled_att_decoder_initial_state = self.tiled_att_decoder_initial_state.clone(
						cell_state=self.tiled_encoder_final_state)

			with tf.name_scope('teacher_forcing'):
				self.decoder_inputs = tf.transpose(self.em_de_inputs, [1, 0, 2], name='decoder_inputs')
				self.train_helper = tf.contrib.seq2seq.TrainingHelper(
					inputs=self.decoder_inputs,
					sequence_length=self.de_lengths,
					time_major = True,
					name='train_decoder_helper')
				self.train_decoder = tf.contrib.seq2seq.BasicDecoder(
					cell=self.att_cell,
					helper=self.train_helper,
					initial_state=self.att_decoder_initial_state,
					output_layer=self.decoder_dense)
				with tf.variable_scope('decode_with_shared_params'):
					self.train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
						decoder=self.train_decoder,
						output_time_major=False,
						impute_finished=True,
						maximum_iterations=self.de_max_len)
				self.train_decoder_logits = self.train_decoder_output.rnn_output
			
			with tf.name_scope('test_decoding_layer'):
				self.beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
					cell=self.tiled_att_cell, 
					embedding=self.de_embedding, 
					start_tokens=self.start_slice, 
					end_token=self.config.end_idx, 
					initial_state= self.tiled_att_decoder_initial_state, 
					beam_width=self.config.beam_width, 
					output_layer=self.decoder_dense, 
					length_penalty_weight=self.config.beam_len_norm, 
					coverage_penalty_weight=self.config.beam_conv_penalty)

				with tf.variable_scope('decode_with_shared_params', reuse=True):
					self.test_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
						decoder=self.beam_search_decoder,
						output_time_major=False,
						impute_finished=False,
						maximum_iterations=self.de_max_len)
					self.test_decoder_preds = tf.unstack(self.test_decoder_output.predicted_ids, axis=2)[0]

		with tf.name_scope('output_layer'):
			self.decoder_preds = tf.argmax(
				self.train_decoder_logits, 
				axis=2, 
				output_type=tf.int32, 
				name='decoder_prediction')
			
		with tf.name_scope('evaluation'):
			self.mask_weights = tf.cast(tf.sequence_mask(self.de_lengths, self.de_max_len), tf.float32)
			self.seq_loss = tf.contrib.seq2seq.sequence_loss(
				logits=self.train_decoder_logits,
				targets=self.y_targets,
				weights=self.mask_weights,
				name='sequence_loss')
			self.loss = tf.reduce_mean(self.seq_loss, name='average_loss')
			self.acc = tf.contrib.metrics.accuracy(
				predictions=self.decoder_preds,
				labels=self.y_targets,
				weights=self.mask_weights,
				name='token_accuracy')
		
		with tf.name_scope('optimization'):
			self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate, decay=self.config.rmsp_decay_rate)
			gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
			gradients = [None if gradient is None else tf.clip_by_norm(gradient, self.config.clipping_threshold) for gradient in gradients]			
			self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)