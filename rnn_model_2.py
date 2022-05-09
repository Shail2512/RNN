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
        self.rnn = torch.nn.RNN(18, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(10, 1)

     def forward(self, inp, age3_unique, age6_unique, age9_unique):
        inp = inp.permute((1,0,2))
        final_out_3 = self.fc3(age3_unique)[None,:,:]
        #Concatenating with original input
        final_input_age3 = torch.cat((inp[0,:,:][None,:,:],final_out_3),2)
        final_input = final_input_age3
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(final_input, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = torch.sigmoid(self.fc2(hidden2))
        final_out_overall = final_out_overall.permute((1,0,2))
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)  
        return final_out_overall.permute(1,0), hn



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
        self.rnn = torch.nn.RNN(18, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(9, 2)
        self.fc6 = torch.nn.Linear(2, 2)
    
     def forward(self, inp, age3_unique, age6_unique, age9_unique):
        inp = inp.permute((1,0,2))
        final_out_3 = self.fc3(age3_unique)[None,:,:]
        final_out_6 = self.fc6(age6_unique)[None,:,:]
        #Concatenating with original input
        final_input_age3 = torch.cat((inp[0,:,:][None,:,:],final_out_3),2)
        final_input_age6 = torch.cat((inp[1,:,:][None,:,:],final_out_6),2)
        final_input = torch.cat((final_input_age3, final_input_age6), dim=0)
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(final_input, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = torch.sigmoid(self.fc2(hidden2))
        final_out_overall = final_out_overall.permute((1,0,2))
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)  
        return final_out_overall.permute(1,0), hn

   
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
        self.rnn = torch.nn.RNN(18, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(9, 2)
        self.fc6 = torch.nn.Linear(2, 2)
        self.fc9 = torch.nn.Linear(4, 2)

    def forward(self, inp, age3_unique, age6_unique, age9_unique):
        inp = inp.permute((1,0,2))
        final_out_3 = self.fc3(age3_unique)[None,:,:]
        final_out_6 = self.fc6(age6_unique)[None,:,:]
        final_out_9 = self.fc9(age9_unique)[None,:,:]
        #Concatenating with original input
        final_input_age3 = torch.cat((inp[0,:,:][None,:,:],final_out_3),2)
        final_input_age6 = torch.cat((inp[1,:,:][None,:,:],final_out_6),2)
        final_input_age9 = torch.cat((inp[2,:,:][None,:,:],final_out_9),2)
        final_input = torch.cat((final_input_age3, final_input_age6, final_input_age9), dim=0)
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(final_input, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = torch.sigmoid(self.fc2(hidden2))
        final_out_overall = final_out_overall.permute((1,0,2))
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)  
        return final_out_overall.permute(1,0), hn

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
        self.rnn = torch.nn.RNN(18, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False)
        
        # fc layers
        self.fc0 = torch.nn.Linear(self.hidden_size, 128)
        self.fc1 = torch.nn.Linear(128, 16)
        self.fc2 = torch.nn.Linear(16, 1)
        #self.fcFinal = torch.nn.Linear(hidden_dim, 4)
        self.fc3 = torch.nn.Linear(9, 2)
        self.fc6 = torch.nn.Linear(2, 2)
        self.fc9 = torch.nn.Linear(4, 2)
        self.fc12 = torch.nn.Linear(3, 2)
    
     def forward(self, inp, age3_unique, age6_unique, age9_unique):
        inp = inp.permute((1,0,2))
        final_out_3 = self.fc3(age3_unique)[None,:,:]
        final_out_6 = self.fc6(age6_unique)[None,:,:]
        final_out_9 = self.fc9(age9_unique)[None,:,:]
        final_out_12 = self.fc12(age12_unique)[None,:,:]

        #Concatenating with original input
        final_input_age3 = torch.cat((inp[0,:,:][None,:,:],final_out_3),2)
        final_input_age6 = torch.cat((inp[1,:,:][None,:,:],final_out_6),2)
        final_input_age9 = torch.cat((inp[2,:,:][None,:,:],final_out_9),2)
        final_input_age12 = torch.cat((inp[3,:,:][None,:,:],final_out_12),2)
 
        final_input = torch.cat((final_input_age3, final_input_age6, final_input_age9, final_input_age12), dim=0)
        # initialize hidden and cell
        hn = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        # step through the sequence i.e. for each wave
        output, hn = self.rnn(final_input, hn.cuda())
        hidden1 = self.fc0(output)
        hidden2 = self.fc1(hidden1)
        final_out_overall = torch.sigmoid(self.fc2(hidden2))
        final_out_overall = final_out_overall.permute((1,0,2))
        final_out_overall = final_out_overall.contiguous().view(final_out_overall.shape[0],-1)  
        return final_out_overall.permute(1,0), hn

