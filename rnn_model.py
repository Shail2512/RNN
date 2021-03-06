import torch
import numpy as np
class RNNModelAge6(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size):
        super(RNNModelAge6, self).__init__()
        

        # RNN Architecture
        self.hidden_size=hidden_dim
        self.input_size=input_dim 
        self.batch_size=batch_size
        self.num_layers=num_layers
        #print("Input dim : ", input_dim," Feature dim: ",feature_dim," hidden dim: ", hidden_dim," output_dim: ",output_dimi)  
        # RNN 
        self.rnn = torch.nn.RNN(self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, inp, age3_unique):
        inp = inp.permute((1,0,2))
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(inp, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = self.fc2(hidden2)
        final_out_overall = final_out_overall.permute((1,0,2))
        #print("Final Output Shape : ", final_out_overall.shape)
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)
        #print("Final Output Overall : ", final_out_overall.shape)
        final_input_age3 = torch.cat((final_out_overall[:,0][:,None],age3_unique),1)
        #final_input_age12 = torch.cat((final_out_overall[:,3][:,None],age12_unique),1)
        final_out_3 = torch.sigmoid(self.fc3(final_input_age3)).permute(1,0)
        #final_out_12 = torch.sigmoid(self.fc12(final_input_age12)).permute(1,0)
        #print("Final Output 12 : ", final_out_12)
        #print(final_out_3.shape, " ", final_out_6.shape, " ")
        #final_output = torch.cat((final_out_3, final_out_6, final_out_9, final_out_12), dim=0)
        final_output = final_out_3
        #final_output = torch.Tensor([final_out_3.cpu().detach().numpy(), final_out_6.cpu().detach().numpy(), final_out_9.cpu().detach().numpy(), final_out_12.cpu().detach().numpy()])
        #final_out = final_out_6.contiguous().view(final_out_6.shape[0],-1)
        
        print("Initial Output: ", output.shape, " Hidden shape: ",hn.shape,"Final_output : " ,final_output.shape)  
        return final_output, hn
class RNNModelAge9(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size):
        super(RNNModelAge9, self).__init__()
        

        # RNN Architecture
        self.hidden_size=hidden_dim
        self.input_size=input_dim 
        self.batch_size=batch_size
        self.num_layers=num_layers
        #print("Input dim : ", input_dim," Feature dim: ",feature_dim," hidden dim: ", hidden_dim," output_dim: ",output_dimi)  
        # RNN 
        self.rnn = torch.nn.RNN(self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(10, 1)
        self.fc6 = torch.nn.Linear(3, 1)

    def forward(self, inp, age3_unique, age6_unique):
        inp = inp.permute((1,0,2))
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(inp, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = self.fc2(hidden2)
        final_out_overall = final_out_overall.permute((1,0,2))
        #print("Final Output Shape : ", final_out_overall.shape)
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)
        #print("Final Output Overall : ", final_out_overall.shape)
        final_input_age3 = torch.cat((final_out_overall[:,0][:,None],age3_unique),1)
        final_input_age6 = torch.cat((final_out_overall[:,1][:,None],age6_unique),1)
        #final_input_age12 = torch.cat((final_out_overall[:,3][:,None],age12_unique),1)
        final_out_3 = torch.sigmoid(self.fc3(final_input_age3)).permute(1,0)
        final_out_6 = torch.sigmoid(self.fc6(final_input_age6)).permute(1,0)
        #final_out_12 = torch.sigmoid(self.fc12(final_input_age12)).permute(1,0)
        #print("Final Output 12 : ", final_out_12)
        #print(final_out_3.shape, " ", final_out_6.shape, " ")
        #final_output = torch.cat((final_out_3, final_out_6, final_out_9, final_out_12), dim=0)
        final_output = torch.cat((final_out_3, final_out_6), dim=0)
        #final_output = torch.Tensor([final_out_3.cpu().detach().numpy(), final_out_6.cpu().detach().numpy(), final_out_9.cpu().detach().numpy(), final_out_12.cpu().detach().numpy()])
        #final_out = final_out_6.contiguous().view(final_out_6.shape[0],-1)
        
        print("Initial Output: ", output.shape, " Hidden shape: ",hn.shape,"Final_output : " ,final_output.shape)  
        return final_output, hn
class RNNModelAge12(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size):
        super(RNNModelAge12, self).__init__()
        

        # RNN Architecture
        self.hidden_size=hidden_dim
        self.input_size=input_dim 
        self.batch_size=batch_size
        self.num_layers=num_layers
        #print("Input dim : ", input_dim," Feature dim: ",feature_dim," hidden dim: ", hidden_dim," output_dim: ",output_dimi)  
        # RNN 
        self.rnn = torch.nn.RNN(self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(10, 1)
        self.fc6 = torch.nn.Linear(3, 1)
        self.fc9 = torch.nn.Linear(5, 1)

    def forward(self, inp, age3_unique, age6_unique, age9_unique):
        inp = inp.permute((1,0,2))
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(inp, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = self.fc2(hidden2)
        final_out_overall = final_out_overall.permute((1,0,2))
        #print("Final Output Shape : ", final_out_overall.shape)
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)
        #print("Final Output Overall : ", final_out_overall.shape)
        final_input_age3 = torch.cat((final_out_overall[:,0][:,None],age3_unique),1)
        final_input_age6 = torch.cat((final_out_overall[:,1][:,None],age6_unique),1)
        final_input_age9 = torch.cat((final_out_overall[:,2][:,None],age9_unique),1)
        #final_input_age12 = torch.cat((final_out_overall[:,3][:,None],age12_unique),1)
        final_out_3 = torch.sigmoid(self.fc3(final_input_age3)).permute(1,0)
        final_out_6 = torch.sigmoid(self.fc6(final_input_age6)).permute(1,0)
        final_out_9 = torch.sigmoid(self.fc9(final_input_age9)).permute(1,0)
        final_output = torch.cat((final_out_3, final_out_6, final_out_9), dim=0)
        #final_output = torch.Tensor([final_out_3.cpu().detach().numpy(), final_out_6.cpu().detach().numpy(), final_out_9.cpu().detach().numpy(), final_out_12.cpu().detach().numpy()])
        #final_out = final_out_6.contiguous().view(final_out_6.shape[0],-1)
        
        print("Initial Output: ", output.shape, " Hidden shape: ",hn.shape,"Final_output : " ,final_output.shape)  
        return final_output, hn
class RNNModelAge15(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_size):
        super(RNNModelAge15, self).__init__()
        

        # RNN Architecture
        self.hidden_size=hidden_dim
        self.input_size=input_dim 
        self.batch_size=batch_size
        self.num_layers=num_layers
        #print("Input dim : ", input_dim," Feature dim: ",feature_dim," hidden dim: ", hidden_dim," output_dim: ",output_dimi)  
        # RNN 
        self.rnn = torch.nn.RNN(self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(10, 1)
        self.fc6 = torch.nn.Linear(3, 1)
        self.fc9 = torch.nn.Linear(5, 1)
        self.fc12 = torch.nn.Linear(4, 1)

    def forward(self, inp, age3_unique, age6_unique, age9_unique,age12_unique):
        inp = inp.permute((1,0,2))
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(inp, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = self.fc2(hidden2)
        final_out_overall = final_out_overall.permute((1,0,2))
        #print("Final Output Shape : ", final_out_overall.shape)
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)
        #print("Final Output Overall : ", final_out_overall.shape)
        final_input_age3 = torch.cat((final_out_overall[:,0][:,None],age3_unique),1)
        final_input_age6 = torch.cat((final_out_overall[:,1][:,None],age6_unique),1)
        final_input_age9 = torch.cat((final_out_overall[:,2][:,None],age9_unique),1)
        final_input_age12 = torch.cat((final_out_overall[:,3][:,None],age12_unique),1)
        final_out_3 = torch.sigmoid(self.fc3(final_input_age3)).permute(1,0)
        final_out_6 = torch.sigmoid(self.fc6(final_input_age6)).permute(1,0)
        final_out_9 = torch.sigmoid(self.fc9(final_input_age9)).permute(1,0)
        final_out_12 = torch.sigmoid(self.fc12(final_input_age12)).permute(1,0)
        #print("Final Output 12 : ", final_out_12)
        #print(final_out_3.shape, " ", final_out_6.shape, " ")
        #final_output = torch.cat((final_out_3, final_out_6, final_out_9, final_out_12), dim=0)
        final_output = torch.cat((final_out_3, final_out_6, final_out_9, final_out_12), dim=0)
        #final_output = torch.Tensor([final_out_3.cpu().detach().numpy(), final_out_6.cpu().detach().numpy(), final_out_9.cpu().detach().numpy(), final_out_12.cpu().detach().numpy()])
        #final_out = final_out_6.contiguous().view(final_out_6.shape[0],-1)
        
        print("Initial Output: ", output.shape, " Hidden shape: ",hn.shape,"Final_output : " ,final_output.shape)  
        return final_output, hn
