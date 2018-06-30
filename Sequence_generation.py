from Char_RNN import Char_RNN


# Load the model
data_file = 'all_hugo.txt'
model_name = 'All_hugo50_3L512n05p.h5'
char_rnn = Char_RNN(data_file, model_name=model_name)



# Generate a sequence
output_file = 'generated_' + data_file
char_rnn.generate_sequence(sequence_length=10000, temperature=0.8, output_file=output_file)