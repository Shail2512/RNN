from dlatk.featureGetter import FeatureGetter
import numpy as np
import torch
import rnn_model
import torch.nn.functional as F
from sklearn import metrics
fg = FeatureGetter.fromFile('./test.ini')
data = fg.getGroupNorms()
np_data = np.array(data)
#print(np_data.shape)
#print(np_data[:652])
data_age_3_unique = ["CBQ_EC_Mother_age3","AnyDepressionDuke_PAPA_age3","DSMAnxDxDuke_PAPA_age3","ODDDxDuke2_PAPA_age3","ADHDdx_PAPA_age3","CBCL_DSMAffectiveProblems_Mother_age3","CBCL_DSMAnxietyProblems_Mother_age3","CBCL_DSMAttentionDeficitHyperactivityProblems_Mother_age3","CBCL_DSMOppositionalDefiantProblems_Mother_age3","Incapacity_MO_PAPA_age3"]
data_age_3_consistent = ["CBQ_NA_Mother_age3","CBQ_ApproachAnticipation_Mother_age3","CBQ_SmilingLaughter_Mother_age3","totalstress.typecount.age3","PSDQ_Factor1_Authoritative_Parent_age3","PSDQ_Factor2_Authoritarian_Parent_age3","PSDQ_Factor3_Permissive_Parent_age3","DyadicAdjust_Parent_Abrev_age3"]
data_age_6_unique = ["CBQ_EC_Mother_age6","LifeRIFT_total_S_age6"]
data_age_6_consistent = ["CBQ_NA_Mother_age6","CBQ_Age6_ApproachAnticipation_Mother","CBQ_Age6_SmilingLaughter_Mother","totalstress.typecount.age6","PSDQ_Age6_Factor1_Authoritative_Parent","PSDQ_Age6_Factor2_Authoritarian_Parent","PSDQ_Age6_Factor3_Permissive_Parent","DyadicAdjust_Age6_Parent_Abrev"]
data_age_9_unique = ["AFARS_NAscale_Child_T3","APP_Total_Child_T3","PDSbothgenders_age9_Mother","LifeRIFT_total_S_age9"]
data_age_9_consistent = ["Eisenberg_ChildReactions_NegativeEmotionsScale_Mother_T3","Eisenberg_ChildReactions_PositiveEmotionsScale_Mother_T3","AFARS_PAscale_Child_T3","Total_Stressors_Trauma_Sum_KSADS9","Acceptance_Rejection_CRPBIc_Parent_age9","Control_Autonomy_CRPBIc_Parent_age9","Firm_LaxControl_CRPBIc_Parent_age9","DAS_Total_age9_Parent"]
data_age_12_unique = ["SNAPY_Disinhibition_Age12.Father","NRI_CLOSENESS_MOTHER_TOTAL_CHILD_Age12"," NRI_CLOSENESS_Father_TOTAL_CHILD_Age12","NRI_DISCORD_MOTHER_TOTAL_CHILD_Age12"," NRI_DISCORD_Father_TOTAL_CHILD_Age12","MSS_Friends_Child_Age12","PDStotal_Child_12"]
data_age_12_consistent = ["SNAPY_NegativeTemperament_Age12.Mother","SNAPY_PositiveTemperament_Age12.Mother","SNAPY_PositiveTemperament_Age12.Mother","LSI.Total.ImpactSum.ModSevere.BeforeKSADS.std_12","Acceptance_Rejection_CRPBI_age12.Parent","Control_Autonomy_CRPBI_age12.Parent","Firm_LaxControl_CRPBI_age12.Parent","DAS_Total_age12.Parent"]
data_constants = ["Sex","Race_updated","Ethnicity_updated","LifetimeMoodDisorderP","LifetimeAnxietyDisorderP","LifetimeSubstanceDisorderP","PPVTScoreStandard_Age6"]
group_id = []
data_age_3_consistent.extend(data_constants)
data_age_6_consistent.extend(data_constants)
data_age_9_consistent.extend(data_constants)
data_age_12_consistent.extend(data_constants)
'''
for i in range(651):
    group_id.append(data[i][0])
np_data_age3_consistent = []
for i in group_id:
    l=[]
    for j in data_age_3_consistent:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
    np_data_age3_consistent.append(np.array(l))
final_data_age3_consistent = np.array(np_data_age3_consistent)
print(final_data_age3_consistent.shape)
print(final_data_age3_consistent[:5])
with open("data_age_3_rnn.npy","wb") as f:
    np.save(f, final_data_age3_consistent)

for i in range(651):
    group_id.append(data[i][0])
np_data_age3_consistent = []
for i in group_id:
    l=[]
    for j in data_age_3_consistent:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
    np_data_age3_consistent.append(np.array(l))
final_data_age3_consistent = np.array(np_data_age3_consistent)
print(final_data_age3_consistent.shape)
print(final_data_age3_consistent[:5])
with open("data_age_3_rnn.npy","wb") as f:
    np.save(f, final_data_age3_consistent)

for i in range(651):
    group_id.append(data[i][0])
np_data_age9_consistent = []
for i in group_id:
    l=[]
    for j in data_age_9_consistent:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
                break
    np_data_age9_consistent.append(np.array(l))
final_data_age9_consistent = np.array(np_data_age9_consistent)
print(final_data_age9_consistent.shape)
print(final_data_age9_consistent[:5])
with open("data_age_9_rnn.npy","wb") as f:
    np.save(f, final_data_age9_consistent)

for i in range(651):
    group_id.append(data[i][0])
np_data_age12_consistent = []
for i in group_id:
    l=[]
    for j in data_age_12_consistent:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
                break
    np_data_age12_consistent.append(np.array(l))
final_data_age12_consistent = np.array(np_data_age12_consistent)
print(final_data_age12_consistent.shape)
print(final_data_age12_consistent[:5])
with open("data_age_12_rnn.npy","wb") as f:
    np.save(f, final_data_age12_consistent)



for i in range(651):
    group_id.append(data[i][0])
np_data_age3_unique = []
for i in group_id:
    l=[]
    for j in data_age_3_unique:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
    np_data_age3_unique.append(np.array(l))
final_data_age3_unique = np.array(np_data_age3_unique)
print("Unique age 3 : ", final_data_age3_unique.shape)
#print(final_data_age3_unique[:5])
with open("data_age_3_rnn_unique.npy","wb") as f:
    np.save(f, final_data_age3_unique)

#for i in range(651):
#    group_id.append(data[i][0])
np_data_age6_unique = []
for i in group_id:
    l=[]
    for j in data_age_6_unique:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
    np_data_age6_unique.append(np.array(l))
final_data_age6_unique = np.array(np_data_age6_unique)
print("Unique age 6 : ", final_data_age6_unique.shape)
#print(final_data_age6_consistent[:5])
with open("data_age_6_rnn_unique.npy","wb") as f:
    np.save(f, final_data_age6_unique)

#for i in range(651):
#    group_id.append(data[i][0])
np_data_age9_unique = []
for i in group_id:
    l=[]
    for j in data_age_9_unique:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
                break
    np_data_age9_unique.append(np.array(l))
final_data_age9_unique = np.array(np_data_age9_unique)
print("Unique age 9: ", final_data_age9_unique.shape)
#print(final_data_age9_unique[:5])
with open("data_age_9_rnn_unique.npy","wb") as f:
    np.save(f, final_data_age9_unique)

#for i in range(651):
#    group_id.append(data[i][0])
np_data_age12_unique = []
for i in group_id:
    l=[]
    for j in data_age_12_unique:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l.append(k[2])
                break
    np_data_age12_unique.append(np.array(l))
final_data_age12_unique = np.array(np_data_age12_unique)
print("Unique age 12: ", final_data_age12_unique.shape)
#print(final_data_age12_unique[:5])
with open("data_age_12_rnn_unique.npy","wb") as f:
    np.save(f, final_data_age12_unique)

'''
with open("data_age_3_rnn.npy","rb") as f:
    data_age_3_rnn = np.load(f)
data_age_3_rnn = np.asfarray(data_age_3_rnn)
data_age_3_rnn = np.reshape(data_age_3_rnn, (1, data_age_3_rnn.shape[0], data_age_3_rnn.shape[1]))
#print(data_age_3_rnn.shape)
with open("data_age_6_rnn.npy","rb") as f:
    data_age_6_rnn = np.load(f)
data_age_6_rnn = np.asfarray(data_age_6_rnn)
data_age_6_rnn = np.reshape(data_age_6_rnn, (1, data_age_6_rnn.shape[0], data_age_6_rnn.shape[1]))
#print(data_age_6_rnn.shape)
with open("data_age_9_rnn.npy","rb") as f:
    data_age_9_rnn = np.load(f)
data_age_9_rnn = np.asfarray(data_age_9_rnn)
data_age_9_rnn = np.reshape(data_age_9_rnn, (1, data_age_9_rnn.shape[0], data_age_9_rnn.shape[1]))
#print(data_age_9_rnn.shape)
with open("data_age_12_rnn.npy","rb") as f:
    data_age_12_rnn = np.load(f)
data_age_12_rnn = np.asfarray(data_age_12_rnn)
data_age_12_rnn = np.reshape(data_age_12_rnn, (1, data_age_12_rnn.shape[0], data_age_12_rnn.shape[1]))
#print(data_age_12_rnn.shape)
with open("data_age_3_rnn_unique.npy","rb") as f:
    data_age_3_rnn_unique = np.load(f)
data_age_3_rnn_unique = np.asfarray(data_age_3_rnn_unique)
data_age_3_rnn_unique = np.reshape(data_age_3_rnn_unique, (1, data_age_3_rnn_unique.shape[0], data_age_3_rnn_unique.shape[1]))
print("Age 3 unique : ", data_age_3_rnn_unique.shape)
with open("data_age_6_rnn_unique.npy","rb") as f:
    data_age_6_rnn_unique = np.load(f)
data_age_6_rnn_unique = np.asfarray(data_age_6_rnn_unique)
data_age_6_rnn_unique = np.reshape(data_age_6_rnn_unique, (1, data_age_6_rnn_unique.shape[0], data_age_6_rnn_unique.shape[1]))
print(data_age_6_rnn_unique.shape)
with open("data_age_9_rnn_unique.npy","rb") as f:
    data_age_9_rnn_unique = np.load(f)
data_age_9_rnn_unique = np.asfarray(data_age_9_rnn_unique)
data_age_9_rnn_unique = np.reshape(data_age_9_rnn_unique, (1, data_age_9_rnn_unique.shape[0], data_age_9_rnn_unique.shape[1]))
print(data_age_9_rnn_unique.shape)
with open("data_age_12_rnn_unique.npy","rb") as f:
    data_age_12_rnn_unique = np.load(f)
data_age_12_rnn_unique = np.asfarray(data_age_12_rnn_unique)
data_age_12_rnn_unique = np.reshape(data_age_12_rnn_unique, (1, data_age_12_rnn_unique.shape[0], data_age_12_rnn_unique.shape[1]))
print(data_age_12_rnn_unique.shape)

output_age_3 = ["AnyDepression_T2_Imp"]
#output_age_3 = ["DSMAnxDx_T2"]

for i in range(651):
    group_id.append(data[i][0])
np_data_age3_output = []
for i in group_id:
    for j in output_age_3:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l = float(k[2])
    np_data_age3_output.append(l)
final_data_age3_output = np.array(np_data_age3_output)

output_age_6 = ["LifeTime_Depression_withNOS_age9"]
#output_age_6 = ["LifeTime_AnyAnxiety_WithoutNOS_age9"]
np_data_age6_output = []
for i in group_id:
    for j in output_age_6:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l = float(k[2])
    np_data_age6_output.append(l)
final_data_age6_output = np.array(np_data_age6_output)

output_age_9 = ["AnyDepressiveDisorder_includingNOS_12"]
#output_age_9 = ["AnyAnxiety12"]
np_data_age9_output = []
for i in group_id:
    for j in output_age_9:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l = float(k[2])
    np_data_age9_output.append(l)
final_data_age9_output = np.array(np_data_age9_output)

output_age_12 = ["AnyDepressiveDisorder_includingNOS_15"]
np_data_age12_output = []
for i in group_id:
    for j in output_age_12:
        for k in np_data:
            if str(k[0]) == str(i) and str(k[1]) == str(j):
                l = float(k[2])
    np_data_age12_output.append(l)
final_data_age12_output = np.array(np_data_age12_output)
print("Data age 9 output_data : ", final_data_age9_output[:20])
#final_data_age6_output = np.reshape(final_data_age6_output, (-1,1))
'''
print(final_data_age3_output.shape)
print(final_data_age3_output[:5])
print(final_data_age6_output.shape)
print(final_data_age6_output[:5])
print(final_data_age9_output.shape)
print(final_data_age9_output[:5])
print(final_data_age12_output.shape)
print(final_data_age12_output[:5])
'''
#s_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
input_data = []
input_data.append(torch.from_numpy(data_age_3_rnn))
input_data.append(torch.from_numpy(data_age_6_rnn))
input_data.append(torch.from_numpy(data_age_9_rnn))
#input_data.append(torch.from_numpy(data_age_12_rnn))

input_data_age3_unique = torch.from_numpy(data_age_3_rnn_unique)
input_data_age6_unique = torch.from_numpy(data_age_6_rnn_unique)
input_data_age9_unique = torch.from_numpy(data_age_9_rnn_unique)
input_data_age12_unique = torch.from_numpy(data_age_12_rnn_unique)

#print("Unique Input Shape : ", X_Unique.shape)
X = torch.Tensor(651,4,15)
torch.cat(input_data, out=X.double())
L = [final_data_age3_output, final_data_age6_output, final_data_age9_output, final_data_age12_output]
output_data = np.vstack(L)

#print("Output data: ", output_data.shape)
Y = torch.from_numpy(output_data)
Y = Y.permute((1,0))
Y = torch.round(Y)
Y_mean = torch.mean(Y, dim=1)
Y_mean = Y_mean.reshape([Y_mean.shape[0],1])
input_data_extend = torch.cat((Y_mean, Y[:,0].reshape([Y.shape[0],1]), Y[:,1].reshape([Y.shape[0],1]), Y[:,2].reshape([Y.shape[0],1])), 1)
input_data_extend = input_data_extend.reshape([input_data_extend.shape[0],input_data_extend.shape[1],1])
X_extend = torch.cat((X.float(),input_data_extend.float()),2)
X_train = X_extend[:585,:,:]
X_test = X_extend[-66:,:,:]
X_Unique_age3_train = input_data_age3_unique.flatten(0,1)[:585,:]
X_Unique_age3_test = input_data_age3_unique.flatten(0,1)[-66:,:]
X_Unique_age6_train = input_data_age6_unique.flatten(0,1)[:585,:]
X_Unique_age6_test = input_data_age6_unique.flatten(0,1)[-66:,:]
X_Unique_age9_train = input_data_age9_unique.flatten(0,1)[:585,:]
X_Unique_age9_test = input_data_age9_unique.flatten(0,1)[-66:,:]
X_Unique_age12_train = input_data_age12_unique.flatten(0,1)[:585,:]
X_Unique_age12_test = input_data_age12_unique.flatten(0,1)[-66:,:]
Y_train = Y[:585,:]
Y_test = Y[-66:, :]
model = rnn_model.RNNModel(X_train.shape[2], X_train.shape[1],256, 4, 2, X_train.shape[0])
model = model.to(device)
print(model)
num_epochs = 100
lr = 0.01
# Define Loss, Optimizer
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#input_seq = input_.to(device)
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    #output, hidden = model(X_train.float().cuda(), X_Unique_age3_train.float().cuda(), X_Unique_age6_train.float().cuda(), X_Unique_age9_train.float().cuda(), X_Unique_age12_train.float().cuda())
    output, hidden = model(X_train.float().cuda(), X_Unique_age3_train.float().cuda(), X_Unique_age6_train.float().cuda(), X_Unique_age9_train.float().cuda())
    output = output.to(device)
    #output = output.permute((1,0))
    #output = output.reshape(output.shape[0])
    target_seq = Y_train.to(device)
    target_seq = target_seq.permute((1,0))
    target_seq = target_seq[:3,:]
    print("Target Sequence : ", target_seq.shape)
    target_array = target_seq.cpu().detach().numpy()
    output_array = output.cpu().detach().numpy()
    print("Target Array : ",len(target_array[2]), " Output array : ", len(output_array[2]))
    fpr, tpr, thresholds = metrics.roc_curve(target_array[2], output_array[2])
    print("AUC :", metrics.auc(fpr, tpr))
    loss = criterion(output, target_seq.float())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%5 == 0:
        print('Epoch: {}/{}.............'.format(epoch, num_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

