import torch
class RNNModel(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, batch_size):
        super(RNNModel, self).__init__()
        
        # RNN Architecture
        self.hidden_size=hidden_dim
        self.input_size=feature_dim  
        self.batch_size=batch_size
        self.num_layers=2
        
        # RNN 
        self.rnn = torch.nn.RNN(feature_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # fc layers
        self.fc6 = torch.nn.Linear(hidden_dim, 2)
        self.fc9 = torch.nn.Linear(hidden_dim, 2)
        self.fc12 = torch.nn.Linear(hidden_dim, 2)
        self.fc15 = torch.nn.Linear(hidden_dim, 2)    
        self.linear_list = [self.fc6,self.fc9,self.fc12,self.fc15]

    def forward(self, inp):
        output_list = []
        # initialize hidden and cell
        hn = torch.nn.init(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        cn = torch.nn.init(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        
        # step through the sequence i.e. for each wave
        for i in range(len(inp)):
            output, (hn,cn) = self.rnn(inp[i], (hn,cn))
            # output is [batch size, timestep = 1, hidden dim]
            output_list.append(self.linear_list[i](output[:,0,:]))
        return output_list
