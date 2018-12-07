from keras.utils.vis_utils import plot_model

from keras.models import *
from keras.layers import *
from keras.layers.convolutional import *
from keras.optimizers import *
from keras_contrib.layers import CRF

from model_handler import *

from keras.layers import concatenate

class CRF_ext(CRF):
    @staticmethod
    def _get_recall(y_true, y_pred, mask, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
            
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        judge_not = K.cast(K.not_equal(y_pred, y_true), K.floatx())
        # tp
        f_y_pred = K.cast(y_pred, K.floatx())
        true_positives = K.sum(K.round(K.clip(judge * f_y_pred, 0, 1)))
        # fn + tp
        possible_positives = K.sum(K.round(K.clip(judge_not*((-f_y_pred)+1), 0, 1))) + true_positives
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    @property
    def viterbi_recall(self):
        def recall(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.viterbi_decoding(X, mask)
            return self._get_recall(y_true, y_pred, mask, self.sparse_target)
        recall.func_name = 'viterbi_recall'
        return recall

    @staticmethod
    def _get_precision(y_true, y_pred, mask, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        judge_not = K.cast(K.not_equal(y_pred, y_true), K.floatx())
        # judge is sum of tp and tn; judge_not is sum of fp and fn
        # tp
        f_y_pred = K.cast(y_pred, K.floatx())
        true_positives = K.sum(K.round(K.clip(judge * f_y_pred, 0, 1)))
        # fp + tp
        predicted_positives = K.sum(K.round(K.clip(judge_not * f_y_pred, 0, 1))) + true_positives
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    @property
    def viterbi_precision(self):
        def precision(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.viterbi_decoding(X, mask)
            return self._get_precision(y_true, y_pred, mask, self.sparse_target)
        precision.func_name = 'viterbi_precision'
        return precision
    
    @staticmethod
    def _get_f1(y_true, y_pred, mask, sparse_target=False):
        y_pred = K.argmax(y_pred, -1)
        if sparse_target:
            y_true = K.cast(y_true[:, :, 0], K.dtype(y_pred))
        else:
            y_true = K.argmax(y_true, -1)
        judge = K.cast(K.equal(y_pred, y_true), K.floatx())
        judge_not = K.cast(K.not_equal(y_pred, y_true), K.floatx())
        # judge is sum of tp and tn; judge_not is sum of fp and fn
        # tp
        f_y_pred = K.cast(y_pred, K.floatx())
        true_positives = K.sum(K.round(K.clip(judge * f_y_pred, 0, 1)))
        # fp + tp
        predicted_positives = K.sum(K.round(K.clip(judge_not * f_y_pred, 0, 1))) + true_positives
        precision = true_positives / (predicted_positives + K.epsilon())
        # fn + tp
        possible_positives = K.sum(K.round(K.clip(judge_not*((-f_y_pred)+1), 0, 1))) + true_positives
        recall = true_positives / (possible_positives + K.epsilon())
        f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
        return f1
    
    @property
    def viterbi_f1(self):
        def f1(y_true, y_pred):
            X = self._inbound_nodes[0].input_tensors[0]
            mask = self._inbound_nodes[0].input_masks[0]
            y_pred = self.viterbi_decoding(X, mask)
            return self._get_f1(y_true, y_pred, mask, self.sparse_target)
        f1.func_name = 'viterbi_f1'
        return f1
        

class ReverseComplementLayer(Layer):
    def __init__(self, seq_begin_idx=0, seq_end_idx=4,**kwargs):
        super(ReverseComplementLayer, self).__init__(**kwargs)
        self.seq_begin_idx = seq_begin_idx
        self.seq_end_idx = seq_end_idx
        
    def build(self, input_shape):
        super(ReverseComplementLayer, self).build(input_shape)
        self.trainable = False
    
    def call(self, input_layer):
        # grab ACGT
        seq_begin_idx, seq_end_idx = self.seq_begin_idx, self.seq_end_idx
        acgt = input_layer[:,:,seq_begin_idx:seq_end_idx]
        rev_complement = K.reverse(K.reverse(acgt, axes=-1), axes=-2)
        # concat the additional info back into reverse complemented sequence
        
        if seq_begin_idx:
            features =  concatenate([input_layer[:,:,:seq_begin_idx],input_layer[:,:,seq_end_idx:]], axis=-2)
        else:
            features = input_layer[:,:,seq_end_idx:]
        rev_order_features = K.reverse(features, axes=-2)
        return concatenate([rev_complement, rev_order_features], axis=-1)
        
    def compute_output_shape(self, input_shape):
        return input_shape

def use_reverse_complement(seq_begin_idx=0, seq_end_idx=4):
    def reverse_complement_f(model_builder):
        def rev_compl_added(input_layer, *args, **kwargs):
            import keras.backend as K
            # assumption 1: one hot of nucleotides are in this order A, C, G, T
            # assumption 2: dimensions (samples, each_nt, features_per_nt)
            
            def lambda_layer(input_layer):
                # grab ACGT
                acgt = input_layer[:,:,seq_begin_idx:seq_end_idx]
                rev_complement = K.reverse(K.reverse(acgt, axes=-1), axes=-2)
                # concat the additional info back into reverse complemented sequence
                from keras.layers import concatenate
                if seq_begin_idx:
                    features =  concatenate([input_layer[:,:,:seq_begin_idx],input_layer[:,:,seq_end_idx:]], axis=-2)
                else:
                    features = input_layer[:,:,seq_end_idx:]
                rev_order_features = K.reverse(features, axes=-2)
                return concatenate([rev_complement, rev_order_features], axis=-1)
                
            return model_builder([input_layer, Lambda(lambda_layer)(input_layer)], *args, **kwargs)
        return rev_compl_added
    return reverse_complement_f

class Models:
    @staticmethod
    def template(input_layer, *parameters):
        '''
        Models should contain input_layer arg in 1st position.
        '''
        raise NotImplementedError
    
    @staticmethod
    def use(generator, model_name=None):
        '''
        Generates a function that accepts the input layer for the model. That function will output a ModelHandler.
        ie use:g->(layer->handler)
        '''
        model = ModelHandler(model_name if model_name else generator.__name__)
        def model_starter(input_layer_shape, *args, **kwargs):
            from functools import partial
            model.model_factory=partial(generator, Input(shape=input_layer_shape, *args, **kwargs))
            model.model_factory.__name__=generator.__name__
            return model        
        return model_starter
    
    
    
    @staticmethod
    def factor_net(input_layer, 
                   cnn_kernel_size=26,
                   cnn_n_filters= 32, 
                   lstm_size= 32,
                   d1_size = 128,
                   do_rate = 0.5,
                   optimizer='adam'):
        
        def build(input_layer, hidden_layers):
            output_layer = input_layer
            for hidden_layer in hidden_layers:
                output_layer = hidden_layer(output_layer)
            return output_layer
        
        w2 = cnn_kernel_size//2
        hidden_layers = [
            Conv1D(cnn_n_filters, cnn_kernel_size, activation='relu'),
            Dropout(0.1),
            TimeDistributed(Dense(cnn_n_filters, activation='relu')),
            MaxPool1D(pool_size=w2, strides=w2),
            Bidirectional(LSTM(lstm_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)),
            Dropout(do_rate),
            Flatten(),
            Dense(d1_size, activation='relu'),
            Dropout(do_rate),
            Dense(1, activation='sigmoid')
        ] 
        
        rev_input_layer = ReverseComplementLayer()(input_layer)
        output_layers = [build(layer, hidden_layers) for layer in [input_layer, rev_input_layer]]
        
        output_layer = Average()(output_layers)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        # compile model
        model.compile( optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def factor_net_rev(input_layer, 
                   cnn_kernel_size=26,
                   cnn_n_filters= 32, 
                   lstm_size= 32,
                   d1_size = 128,
                   do_rate = 0.5,
                   optimizer='adam'):
        
        def build(input_layer, hidden_layers):
            output_layer = input_layer
            for hidden_layer in hidden_layers:
                output_layer = hidden_layer(output_layer)
            return output_layer
        
        w2 = cnn_kernel_size//2
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        x = Conv1D(cnn_n_filters, cnn_kernel_size, activation='relu')(base_input)
        x = Dropout(0.1)(x)
        x = TimeDistributed(Dense(cnn_n_filters, activation='relu'))(x)
        x = MaxPool1D(pool_size=w2, strides=w2)(x)
        x = Bidirectional(LSTM(lstm_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
        x = Dropout(do_rate)(x)
        x = Flatten()(x)
        x = Dense(d1_size, activation='relu')(x)
        x = Dropout(do_rate)(x)
        base_out = Dense(1, activation='sigmoid')(x)
        base_model = Model(base_input, base_out)
        
        rev_input_layer = ReverseComplementLayer()(input_layer)
        out, out_rev = base_model(input_layer), base_model(rev_input_layer)
        
        output_layer = Average()([out, out_rev])
        
        model = Model(inputs=input_layer, outputs=output_layer)
        # compile model
        model.compile( optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def bilstm_lstm_tdd_crf(input_layer, 
                            lstm1_size=16, lstm2_size=16, 
                            rec_dropout=0.0,
                            td_dense_size=16, 
                            optimizer='adam'):
        
        bilstm = Bidirectional(LSTM(lstm1_size, 
                                     return_sequences=True, 
                                     recurrent_dropout=rec_dropout))(input_layer)
        
        lstm = LSTM(lstm2_size, stateful=False, return_sequences=True)(bilstm)
        tdd = TimeDistributed(Dense(td_dense_size))(lstm)
        crf_layer = CRF(4)
        crf = crf_layer(tdd)
        model = Model(inputs=input_layer, outputs=crf)
        # compile model
        model.compile( optimizer=optimizer, loss=crf_layer.loss_function , metrics=[crf_layer.accuracy])
        return model
    
#     @staticmethod
#     def s_cnn_bilstm_crf(input_layer, 
#                          cnn_kernel_size=26, cnn_n_filters= 32, 
#                          lstm_size= 256, bi_do_rate = 0.1, rec_do_rate = 0.1,
#                          optimizer='adam'):
        
#         def build(input_layer, hidden_layers):
#             output_layer = input_layer
#             for hidden_layer in hidden_layers:
#                 output_layer = hidden_layer(output_layer)
#             return output_layer
        
#         w2 = cnn_kernel_size//2
#         hidden_layers = [
#             Conv1D(cnn_n_filters, cnn_kernel_size, activation='relu'),
#             Dropout(0.1),
#             TimeDistributed(Dense(cnn_n_filters, activation='relu')),
#             MaxPool1D(pool_size=w2, strides=w2),
#             Bidirectional(LSTM(lstm_size, dropout=0.1, recurrent_dropout=rec_do_rate, return_sequences=True)),
#         ] 
        
#         rev_input_layer = ReverseComplementLayer()(input_layer)
#         output_layers = [build(layer, hidden_layers) for layer in [input_layer, rev_input_layer]]
#         average_layer = Average()(output_layers)
#         crf_layer = CRF(4)
#         crf = crf_layer(average_layer)
        
#         model = Model(inputs=input_layer, outputs=crf)
#         # compile model
#         model.compile( optimizer=optimizer, loss=crf_layer.loss_function , metrics=[crf_layer.accuracy])
#         return model
    
    @staticmethod
    def s_bilstm_crf(input_layer,
                     lstm_size= 64, 
                     bi_do_rate=0.5, rec_do_rate=0.5, 
                     optimizer='adam'):
        
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        lstm = Bidirectional(LSTM(lstm_size, 
                                  dropout=bi_do_rate, 
                                  recurrent_dropout=rec_do_rate, 
                                  return_sequences=True))(base_input)
        base_out = TimeDistributed(Dense(4))(lstm)
        base_model = Model(base_input, base_out)
        base_model.summary()
        rev_input_layer = ReverseComplementLayer()(input_layer)
        out, out_rev = base_model(input_layer), base_model(rev_input_layer)
        
        output_layer = Maximum()([out, out_rev])
        
        crf_layer = CRF_ext(4, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        model.summary()
        # compile model            
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def s_bigru_crf(input_layer,
                     lstm_size= 64, 
                     bi_do_rate=0.5, rec_do_rate=0.5, 
                     optimizer='adam'):
        
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        lstm = Bidirectional(GRU(lstm_size, 
                                  dropout=bi_do_rate, 
                                  recurrent_dropout=rec_do_rate, 
                                  return_sequences=True))(base_input)
        base_out = TimeDistributed(Dense(4))(lstm)
        base_model = Model(base_input, base_out)
        base_model.summary()
        rev_input_layer = ReverseComplementLayer()(input_layer)
        out, out_rev = base_model(input_layer), base_model(rev_input_layer)
        
        output_layer = Maximum()([out, out_rev])
        
        crf_layer = CRF_ext(4, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        model.summary()
        # compile model            
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def s_lstm_crf(input_layer,
                     lstm_size= 64, 
                     bi_do_rate=0.5, rec_do_rate=0.5, 
                     optimizer='adam'):
        
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        lstm = LSTM(lstm_size, 
                                  dropout=bi_do_rate, 
                                  recurrent_dropout=rec_do_rate, 
                                  return_sequences=True)(base_input)
        base_out = TimeDistributed(Dense(4))(lstm)
        base_model = Model(base_input, base_out)
        rev_input_layer = ReverseComplementLayer()(input_layer)
        out, out_rev = base_model(input_layer), base_model(rev_input_layer)
        
        output_layer = Maximum()([out, out_rev])
        
        crf_layer = CRF_ext(4, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        # compile model            
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def bilstm_crf(input_layer,
                     lstm_size= 64, 
                     bi_do_rate=0.5, rec_do_rate=0.5, 
                     optimizer='adam'):
        
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        lstm = Bidirectional(LSTM(lstm_size, 
                                  dropout=bi_do_rate, 
                                  recurrent_dropout=rec_do_rate, 
                                  return_sequences=True))(base_input)
        base_out = TimeDistributed(Dense(4))(lstm)
        base_model = Model(base_input, base_out)
        
        out = base_model(input_layer)
        
        output_layer = out
        
        crf_layer = CRF_ext(4, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        # compile model            
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def bigru_crf(input_layer,
                     lstm_size= 64, 
                     bi_do_rate=0.5, rec_do_rate=0.5, 
                     optimizer='adam'):
        
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        lstm = Bidirectional(GRU(lstm_size, 
                                  dropout=bi_do_rate, 
                                  recurrent_dropout=rec_do_rate, 
                                  return_sequences=True))(base_input)
        base_out = TimeDistributed(Dense(4))(lstm)
        base_model = Model(base_input, base_out)
        
        out = base_model(input_layer)
        
        output_layer = out
        
        crf_layer = CRF_ext(4, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        # compile model            
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def s_cnn_bilstm_crf(input_layer, 
                   cnn_kernel_size=26, cnn_n_filters= 32, cnn_do_rate=0.1, 
                   lstm_size= 32,
                   d1_size = 128,
                   do_rate = 0.5,
                   crf_size=4,
                   optimizer='adam'):
        
        def build(input_layer, hidden_layers):
            output_layer = input_layer
            for hidden_layer in hidden_layers:
                output_layer = hidden_layer(output_layer)
            return output_layer
        
        w2 = cnn_kernel_size//2
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        x = Conv1D(cnn_n_filters, cnn_kernel_size, activation='relu', padding='same')(base_input)
        x = Dropout(cnn_do_rate)(x)
        x = TimeDistributed(Dense(cnn_n_filters, activation='relu'))(x)
        x = Bidirectional(LSTM(lstm_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
        base_model = Model(base_input, x)
        
        rev_input_layer = ReverseComplementLayer()(input_layer)
        out, out_rev = base_model(input_layer), base_model(rev_input_layer)
        
        output_layer = Maximum()([out, out_rev])
        
        crf_layer = CRF_ext(crf_size, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        # compile model
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def s_cnn_bigru_crf(input_layer, 
                   cnn_kernel_size=26, cnn_n_filters= 32, cnn_do_rate=0.1, 
                   rnn_size= 32,
                   d1_size = 128,
                   do_rate = 0.5,
                   crf_size=4,
                   optimizer='adam'):
        
        def build(input_layer, hidden_layers):
            output_layer = input_layer
            for hidden_layer in hidden_layers:
                output_layer = hidden_layer(output_layer)
            return output_layer
        
        w2 = cnn_kernel_size//2
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        x = Conv1D(cnn_n_filters, cnn_kernel_size, activation='relu', padding='same')(base_input)
        x = Dropout(cnn_do_rate)(x)
        x = TimeDistributed(Dense(cnn_n_filters, activation='relu'))(x)
        x = Bidirectional(GRU(rnn_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
        base_model = Model(base_input, x)
        
        rev_input_layer = ReverseComplementLayer()(input_layer)
        out, out_rev = base_model(input_layer), base_model(rev_input_layer)
        
        output_layer = Maximum()([out, out_rev])
        
        crf_layer = CRF_ext(crf_size, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        # compile model
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def cnn_bilstm_crf(input_layer, 
                   cnn_kernel_size=26, cnn_n_filters= 32, cnn_do_rate=0.1, 
                   lstm_size= 32,
                   d1_size = 128,
                   do_rate = 0.5,
                   crf_size=4,
                   optimizer='adam'):
        
        def build(input_layer, hidden_layers):
            output_layer = input_layer
            for hidden_layer in hidden_layers:
                output_layer = hidden_layer(output_layer)
            return output_layer
        
        w2 = cnn_kernel_size//2
        base_input = Input(shape=input_layer.shape.as_list()[1:])
        x = Conv1D(cnn_n_filters, cnn_kernel_size, activation='relu', padding='same')(base_input)
        x = Dropout(cnn_do_rate)(x)
        x = TimeDistributed(Dense(cnn_n_filters, activation='relu'))(x)
        x = Bidirectional(LSTM(lstm_size, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(x)
        base_model = Model(base_input, x)
        
        out = base_model(input_layer)
        
        output_layer = out
        
        crf_layer = CRF_ext(crf_size, test_mode='viterbi')
        crf = crf_layer(output_layer)
        
        model = Model(inputs=input_layer, outputs=crf)
        # compile model
        model.compile(optimizer=optimizer, 
                      loss=crf_layer.loss_function , 
                      metrics=[crf_layer.accuracy, crf_layer.viterbi_recall, crf_layer.viterbi_precision, crf_layer.viterbi_f1])
        return model
    
    @staticmethod
    def cnn_cnn(input_layer,
                cnn1_n_filters=3, cnn1_kernel_size=8,
                cnn2_n_filters=3, cnn2_kernel_size=None,
                do_rate=0.2,
                dense_size=4,
                optimizer='adam'):
        
        cnn1 = Conv1D(cnn1_n_filters, 
                      cnn1_kernel_size, 
                      activation='relu')(input_layer)
        pool1 = MaxPool1D(2)(cnn1)
        
        cnn2 = Conv1D(cnn2_n_filters, 
                      cnn2_kernel_size if cnn2_kernel_size else cnn1_kernel_size//2,
                      activation='relu')(pool1)
        pool2 = MaxPool1D(2)(cnn2)
        do = Dropout(do_rate)(pool2)

        flat = Flatten()(do)
        
        d1 = Dense(dense_size, activation='relu')(flat)
        dense = Dense(1, activation='sigmoid')(d1)
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def cnn(input_layer,
                cnn1_n_filters=3, cnn1_kernel_size=8,
                do_rate=0.2,
                dense_size=4,
                optimizer='adam'):
        
        cnn1 = Conv1D(cnn1_n_filters, 
                      cnn1_kernel_size, 
                      activation='relu')(input_layer)
        pool1 = MaxPool1D(2)(cnn1)
        do = Dropout(do_rate)(pool1)
        flat = Flatten()(do)
        
        d1 = Dense(dense_size, activation='relu')(flat)
        dense = Dense(1, activation='sigmoid')(d1)
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def cnn_cnn_lstm(input_layer,
                     cnn1_n_filters=3, cnn1_kernel_size=8, 
                     cnn2_n_filters=3, cnn2_kernel_size=None,
                     do_rate=0.2, 
                     lstm_size=16, 
                     optimizer='adam'):
        
        cnn1 = Conv1D(cnn1_n_filters, 
              cnn1_kernel_size, 
              activation='relu')(input_layer)
        pool1 = MaxPool1D(2)(cnn1)
        
        cnn2 = Conv1D(cnn2_n_filters, 
                      cnn2_kernel_size if cnn2_kernel_size else cnn1_kernel_size//2,
                      activation='relu')(pool1)
        pool2 = MaxPool1D(2)(cnn2)
        
        do = Dropout(do_rate)(pool2)
        
        lstm1 = Bidirectional(LSTM(lstm_size))(do)
        dense = Dense(1, activation='sigmoid')(lstm1)
        
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def cnn_lstm(input_layer,
                     cnn1_n_filters=3, cnn1_kernel_size=8, 
                     cnn2_n_filters=3, cnn2_kernel_size=None,
                     do_rate=0.2, 
                     lstm_size=16, 
                     optimizer='adam'):
        
        cnn1 = Conv1D(cnn1_n_filters, 
              cnn1_kernel_size, 
              activation='relu')(input_layer)
        pool1 = MaxPool1D(2)(cnn1)
        
        cnn2 = Conv1D(cnn2_n_filters, 
                      cnn2_kernel_size if cnn2_kernel_size else cnn1_kernel_size//2,
                      activation='relu')(pool1)
        pool2 = MaxPool1D(2)(cnn2)
        
        do = Dropout(do_rate)(pool2)
        
        lstm1 = Bidirectional(LSTM(lstm_size))(do)
        dense = Dense(1, activation='sigmoid')(lstm1)
        
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def cnn_cnn_bilstm(input_layer,
                       cnn1_n_filters=3, cnn1_kernel_size=8,
                       cnn2_n_filters=3, cnn2_kernel_size=None,
                       do_rate=0.2, 
                       lstm_size=16, 
                       optimizer='adam'):
        
        cnn1 = Conv1D(cnn1_n_filters, 
              cnn1_kernel_size, 
              activation='relu')(input_layer)
        pool1 = MaxPool1D(2)(cnn1)
        
        cnn2 = Conv1D(cnn2_n_filters, 
                      cnn2_kernel_size if cnn2_kernel_size else cnn1_kernel_size//2,
                      activation='relu')(pool1)
        pool2 = MaxPool1D(2)(cnn2)
        
        do = Dropout(do_rate)(pool2)
        
        bilstm1 = LSTM(lstm_size)(do)
        dense = Dense(1, activation='sigmoid')(bilstm1)
        
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def cnn_cnn_dnn(input_layer,
                    cnn1_n_filters=3, cnn1_kernel_size=8, 
                    cnn2_n_filters=3, cnn2_kernel_size=None,
                    do_rate=0.2,
                    d1_size=16, d2_size=8, d3_size=4, 
                    optimizer='adam'):
        cnn1 = Conv1D(cnn1_n_filters, 
              cnn1_kernel_size, 
              activation='relu')(input_layer)
        pool1 = MaxPool1D(2)(cnn1)
        
        cnn2 = Conv1D(cnn2_n_filters, 
                      cnn2_kernel_size if cnn2_kernel_size else cnn1_kernel_size//2,
                      activation='relu')(pool1)
        pool2 = MaxPool1D(2)(cnn2)
        
        do = Dropout(do_rate)(pool2)
        
        flat = Flatten()(do)
        
        d1 = Dense(d1_size)(flat)
        leaky1 = LeakyReLU()(d1)
        d2 = Dense(d2_size)(leaky1)
        leaky2 = LeakyReLU()(d2)
         
        dense = Dense(1, activation='sigmoid')(leaky2)
        
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def bilstm(input_layer, 
               lstm_size=16,rec_dropout=0.2,
               dense_size=16, 
               optimizer='adam'):
        bi = Bidirectional(LSTM(lstm_size, 
                                     return_sequences=True, 
                                     recurrent_dropout=rec_dropout))(input_layer)
        flat = Flatten()(bi)
        dense1 = Dense(dense_size)(flat)
        dense = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1())(dense1)
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    @staticmethod
    def bilstm_cnn_bilstm(input_layer,
                          lstm1_size=16, lstm2_size=16, 
                          rec_dropout=0.0,
                          cnn1_n_filters=3, cnn1_kernel_size=8, 
                          do_rate=0.2, 
                          optimizer='adam'):
        lstm1 = Bidirectional(LSTM(lstm1_size, return_sequences=True, 
                                     recurrent_dropout=rec_dropout))(input_layer)
        cnn1 = Conv1D(cnn1_n_filters, cnn1_kernel_size, activation='relu', strides=2)(lstm1)
        pool1 = MaxPool1D(2)(cnn1)
        do = Dropout(do_rate)(pool1)
        
        lstm2 = Bidirectional(LSTM(lstm2_size, 
                                     recurrent_dropout=rec_dropout))(do)
        
        dense = Dense(1, activation='sigmoid')(lstm2)
        
        model = Model(inputs=input_layer, outputs=dense)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model