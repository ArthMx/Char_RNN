from Char_RNN import Char_RNN

# Set data file and the architecture of the model
data_file = 'nietzsche.txt'
seq_length = 100
n_L = 2
n_nodes = 512
p_dropout = 0.5

char_rnn = Char_RNN(data_file, seq_length=seq_length, n_L=n_L, n_nodes=n_nodes,
                    p_dropout=p_dropout)

# Train the model
char_rnn.train_model(epochs=20, epoch_split=1, batch_size=100, train_ratio=0.95)