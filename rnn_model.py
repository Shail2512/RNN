import torch
class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, num_layers, batch_size):
        super(RNNModel, self).__init__()
        
        # RNN Architecture
        self.hidden_size=hidden_dim
        self.input_size=feature_dim  
        self.batch_size=batch_size
        self.num_layers=num_layers
        
        # RNN 
        self.rnn = torch.nn.RNN(feature_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # fc layers
        self.fc6 = torch.nn.Linear(hidden_dim, 1)
        self.fc9 = torch.nn.Linear(hidden_dim, 1)
        self.fc12 = torch.nn.Linear(hidden_dim, 1)
        self.fc15 = torch.nn.Linear(hidden_dim, 1)    
        self.linear_list = [self.fc6,self.fc9,self.fc12,self.fc15]

    def forward(self, inp):
        output_list = []
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        # cn = torch.nn.init(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(inp, hn.float().cuda())
        # output is [batch size, timestep = 1, hidden dim]
        # output_list.append(self.linear_list[i](output[:,0,:]))
        print("Output : ", output.shape)
        print("Hidden : ", hn.shape)
        # out = output[:,:,-1]
        # out = torch.reshape(out[0], (out[0].shape[0],1))
        print(output.shape)
        final_out = self.fc6(output[0])
        print("Final_output : " ,final_out.shape, " final_output : ", final_out[:5])  
        return final_out, hn
'''
	def init_hidden(self, batch_size):
        	# This method generates the first hidden state of zeros which we'll use in the forward pass
        	hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
         	# We'll send the tensor holding the hidden state to the device we specified earlier as well
        	return hidden
'''
