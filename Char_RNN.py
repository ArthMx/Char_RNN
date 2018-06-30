import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import sys
import random
import os

class Char_RNN():
    '''
    Let you train a RNN on any text file and then generate new text using the RNN.
    ---------------
    Input
            - data_file : name of the txt to use to train the model, also required
                        to generate new sequence of data. The file must be located 
                        in the "./data/" subfolder.
            - model_name : Name of the file where the model will be saved or name 
                           of the file where the model will be loaded, if the file 
                           already exists.
            - seq_length : Length of the sequence of data to be used to do
                            the prediction using the RNN.
            - lower_case : if True, transform all the text to lower_case (reduce dimensionnality)
            - n_L : Number of hidden LSTM layers
            - n_nodes : Number of neurons for each LSTM layers
            - p_dropout : Probability of dropout (there is a Dropout layer after each hidden layers)
            
    Method
            - train_model : to train the model.
            - generate_sequence : to generate new sequence of text.
    
    '''
    def __init__(self, data_file, model_name=None, seq_length=50, lower_case=False, 
                 n_L=3, n_nodes=512, p_dropout=0.5):
        
        # Name of the data file
        self.data_file = './data/' + data_file
        
        # seq_length : number of character taken into account to predict the 
        # next character
        self.seq_length = seq_length
        self.lower_case = lower_case
        self.n_L = n_L
        self.n_nodes = n_nodes
        self.p_dropout = p_dropout
        
        if model_name is None:
            # Name of the file where the model will be saved
            model_name = data_file.split('.')[0]
            if not lower_case:
                model_name = model_name[0].upper() + model_name[1:]
            model_name += '{}_{}L{}n'.format(self.seq_length, self.n_L, self.n_nodes) + \
                            '0' + str(self.p_dropout)[2:] + 'p.h5'
        
        path = './model_saved/'
        
        if not os.path.isdir(path):
            os.mkdir(path)
        self.model_file = path + model_name
        
        # verification variable used as condition
        self.training_data_generator = None
        self.train_model_call = 0
        
        self.load_dataset() # load the data file in self.data varaible
        self.make_dictionaries() # make dictionnary to convert the text to vector form
        
        if os.path.isfile(self.model_file):
            self.load_model() # load existing model
            self.model.summary() # print details of the model
            self.batch_size = int(self.model.get_input_at(0).shape[0])
            self.seq_length = int(self.model.get_input_at(0).shape[1])
            self.n_char = int(self.model.get_input_at(0).shape[2])
            self.n_L = len(self.model.layers)//2
            self.n_nodes = int(self.model.layers[0].output.shape[-1])
            
            if self.model_file[14].isupper():
                self.lower_case = False
            else:
                self.lower_case = True
        else:
            self.model = None # Model will be defined when calling train_model() method
        
        self.pred_model = None # prediction model will be created when generate_sequence() is called
        
        if os.path.isfile(self.model_file) and (self.n_char != int(self.model.get_input_at(0).shape[2])):
            print('Changing lower_case to ' + str(not self.lower_case) + ' to avoid problem of input shape')
            self.lower_case = not self.lower_case
            self.load_dataset()
            self.make_dictionaries()
        
    def load_dataset(self):
        '''Load Dataset'''
        with open(self.data_file, 'r') as f:
            self.data = f.read()
            
        if self.lower_case:
            self.data = self.data.lower()
            
        # Separate the data into paragraph, to use random paragraph as seed to generate new sequences
        self.paragraph = self.data.split(sep='\n')
    
    
    def make_dictionaries(self):
        '''
        make dictionaries to go from string to integer indexing and the other way-
        around for each character.
        '''
        chars = sorted(list(set(self.data)))
        self.n_char = len(chars)
        
        # Dicts to convert string character to numeric value and vice versa
        self.char_to_idx = {char:idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx:char for idx, char in enumerate(chars)}
    
        print('Total number of characters in the data file :', len(self.data))
        print('Number of different characters :', self.n_char)
        print(' '.join(chars))
    
    def one_hot_encode_data(self):
        '''One-hot encode the data (from string to matrix form)'''
        data_encoded = np.zeros((len(self.data), self.n_char), dtype=np.bool)
        
        for i, char in enumerate(self.data):
            data_encoded[i, self.char_to_idx[char]] = 1
        return data_encoded
        
    def data_generator(self, data):
        '''Generate batch of data'''
        n_batch = len(data)//self.batch_size # number of batch per epoch
        n_split = self.batch_size//self.seq_length # number of data split to distribute it into each batch
        split_len = len(data)//(n_split) # number of character separating each split
        
        while True:
            for i in range(0, n_batch):
                # choose batch index
                batch_idx = []
                for j in range(0, n_split):
                    offset = j * split_len
                    batch_idx += list(range(offset + (i * self.seq_length), offset + (i + 1) * self.seq_length))
                
                yield self.make_batch(batch_idx)
                
    def make_batch(self, batch_idx):
        x = []
        y = []
        for i in batch_idx:
            x.append(self.data_encoded[i:i+self.seq_length])
            y.append(self.data_encoded[i+self.seq_length])
        
        return np.array(x), np.array(y)
    
    def load_model(self):
        '''Load a model already trained'''
        print('Loading model...')
        self.model = keras.models.load_model(self.model_file)
        print('Model loaded from :', self.model_file)
        
    def make_new_model(self, batch_size):
        '''Define and return a new model'''
        model = Sequential()
        
        # First layer
        if self.n_L == 1:
            model.add(LSTM(self.n_nodes, return_sequences=False, stateful=True, 
                           batch_input_shape=(batch_size, self.seq_length, self.n_char)))
        if self.n_L > 1:
            model.add(LSTM(self.n_nodes, return_sequences=True, stateful=True, 
                           batch_input_shape=(batch_size, self.seq_length, self.n_char)))
        
        model.add(Dropout(self.p_dropout))
        
        # 2 to L-1 layers
        for l in range(2, self.n_L + 1):
            return_sequences = True
            if l == self.n_L:
                return_sequences = False
            model.add(LSTM(self.n_nodes, return_sequences=return_sequences, stateful=True))
            model.add(Dropout(self.p_dropout))
        
        # Last layer
        model.add(Dense(self.n_char, activation='softmax'))
        
        return model
    
    def train_model(self, epochs=1, epoch_split=1, batch_size=100, train_ratio=0.95):
        '''
        Train the model.
        -------------
        Input 
                - epochs : number of epoch for training
                - batch_size : batch size for training, higly recommended to use
                               a multiple of seq_length, if not, some data won't be used.
                - epoch_split : int, to split the dataset and make each epoch shorter.
                - train_ratio : fraction of data used to trained, the remaining
                                data is used as validation
                                If = 1, no validation set is made.
        
        '''
        if self.model is None:
            self.batch_size = batch_size
            self.model = self.make_new_model(self.batch_size) 
            self.model.compile('adam', loss='categorical_crossentropy')
            self.model.summary() # print details of the model
            
        # change input shape of the model in case a new batch_size is passed
        elif self.batch_size != batch_size:
            # copy the weights of the model to pred_model 
            model_weights = self.model.get_weights()
            
            self.batch_size = batch_size
            self.model = self.make_new_model(self.batch_size) # redifine model with new batch_size
            
            self.model.set_weights(model_weights)
            self.model.compile('adam', loss='categorical_crossentropy')
        
        # Encode the data at the first call of train_model
        if self.train_model_call == 0 or train_ratio != self.train_ratio:
            self.data_encoded = self.one_hot_encode_data()
            self.m = int(train_ratio * len(self.data_encoded))            
            
        self.train_model_call += 1
        self.train_ratio = train_ratio
        
        # Set training data generator and number of step between each validation evaluation
        training_data_generator = self.data_generator(self.data_encoded[:self.m])
        steps_per_epoch = (len(self.data_encoded[:self.m]) // self.batch_size) // epoch_split
        
        # Set validation data generator and number of step to go through the whole val set
        if train_ratio < 1:
            val_data_generator = self.data_generator(self.data_encoded[self.m:])
            validation_steps = len(self.data_encoded[self.m:]) // (self.batch_size)
            monitor = 'val_loss'
            
        # if train_ratio == 1, no validation set is made
        elif train_ratio == 1:
            val_data_generator = None
            validation_steps = None
            monitor = 'loss'
            
        checkpoint = keras.callbacks.ModelCheckpoint(self.model_file, monitor=monitor, save_best_only=True, verbose=1)
        
        # Train the model with split
        for epoch in range(epochs):
            print()
            print('Real Epoch count : {}/{}'.format(epoch+1, epochs))
            print()
            initial_epoch = (epoch) * epoch_split
            final_epoch = (epoch+1) * epoch_split
            self.model.reset_states() # reset the states between each epoch
            self.model.fit_generator(training_data_generator, steps_per_epoch=steps_per_epoch, 
                                     epochs=final_epoch, initial_epoch=initial_epoch,
                                     callbacks=[checkpoint], validation_data=val_data_generator, 
                                     validation_steps=validation_steps)
                
    def sample_random_sentence_seed(self):
        '''Return a random sequence which finish by "/n"'''
        min_len = 0
        min_idx = 0
        while min_len < self.seq_length:
            min_len += len('\n'.join(self.paragraph[:min_idx]))
            min_idx += 1
        
        rnd_pararaph_idx = random.choice(list(range(min_idx, len(self.paragraph))))
        sentence_seed = self.paragraph[rnd_pararaph_idx]
        while len(sentence_seed) < (self.seq_length - 2):
            sentence_seed = self.paragraph[rnd_pararaph_idx - 1] + sentence_seed
            rnd_pararaph_idx -= 1
        sentence_seed += '\n'
        return sentence_seed[-self.seq_length:]
    
    def sample(self, preds):
        '''helper function to sample an index from a probability array'''
        preds = np.log(preds + 1e-8) / self.temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(list(range(self.n_char)), p=preds)
    
    def generate_next_char(self, sentence_seed):
        '''Generate next character of a sequence using the probabilitiy predicted by the model'''
        x = np.zeros((1, self.seq_length, self.n_char))
        for t, char in enumerate(sentence_seed):
            x[0, t, self.char_to_idx[char]] = 1
        
        preds = self.pred_model.predict(x, verbose=0)[0]
        next_index = self.sample(preds)
        
        return self.idx_to_char[next_index]
    
    def generate_sequence(self, sequence_length=400, temperature=1.0, output_file=None):
        '''
        Generate a sequence of characters.
        ----------------------
        Input : 
                - sequence_length : number of charcter for the generated sequence
                - temperature : factor dividing the score computed during prediction,
                                if <1 : characters generated with more confidence,
                                if >1 : characters generated more randomly.
                                a value of [0.5, 0.8] usually works best.
                - output_file : If None, print in real time the sequence
                                if name of a file (.txt), write the generated sequence in the file
        '''
        if self.pred_model is None:
            # create model for prediction using a batch_size of 1
            self.pred_model = self.make_new_model(1)
            
            # copy the weights of the model to pred_model 
            model_weights = self.model.get_weights()
            self.pred_model.set_weights(model_weights)
        self.pred_model.reset_states()
        
        self.temperature = temperature
        # sample a random sequence from the texte in the data file, to use as a seed
        # for the sequence generation
        sentence = self.sample_random_sentence_seed()
        
        if output_file == None:
            print('Generating from: ')
            print(sentence)
            print('Generated text: ')
    
            for i in range(sequence_length):
                next_char = self.generate_next_char(sentence)
                sentence = sentence[1:] + next_char
    
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
        
        else:
            print('Generating from: ')
            print(sentence)
            print('Generating and writing the sequence...')
            generated = ''
            for i in range(sequence_length):
                next_char = self.generate_next_char(sentence)
                sentence = sentence[1:] + next_char
                generated += next_char
                with open(output_file, 'w') as f:
                    f.write(generated)
        