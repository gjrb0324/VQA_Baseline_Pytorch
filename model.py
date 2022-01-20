import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
import torchvision
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from tqdm import tqdm

#Set this result of LSTM  as question feature
class LSTM(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        #I can't understand why embeddign has 'padding_idx' param
        #Guess(In Korean): 문장별 단어수(길이)가 다 달라질수 있어서 padded_sequence로 만들어줘야되는데
        #이때 index 0을 불용어 쓰레기통 같은거로 써서 1부터 쓸모있는 값 취급하려고 한게 아닐가
        #라고 생각했는데 밑에서 xavier_초기화 해줘서 padding_idx무쓸모, 대체 왜 쓴거지?
        self.embedding = nn.Embedding(num_tokens, 300)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(300,1024)
        #Embedding weight 초기화
        init.xavier_uniform(self.embedding.weight)
        
        #위의 CNN에서와 같이 가중치를 xavier로, bias를 0으로 초기화
        #lstm.weight_ih_l0 : 0번째(1번쨰) layer의 input-hidden의 가중치 벡터 -> W_(ii,if,ig,io)
        #lstm.weight_hh_l0 : 0번째(1번째) layer의 hidden-hidden의 가중치 벡터 - > W_(hi,hf,hg,ho)
        #W_(i,f,g,o) : 각각 input, forget, cell, cell, output gate의 가중치들
        #원래 [[i_1],[i_2]...[i_n]],[[f_1],[f_2]...[f_n]],[[g_1]..]..[..[o_n]],
        #([i_1]...[o_n]은 모두 벡터)이던 애를 
        #chunk로 [i(m*n)], [f(m*n)],[g(m*n)],[o(m*n)]로 묶어내서 같이 xavier 초기화
        for w in self.lstm.weight_ih_l0.chunk(4,0):
            init.xavier_uniform_(w)
        for w in self.lstm.weight_hh_l0.chunk(4,0):
            init.xavier_uniform_(w)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
    def forward(self,q,q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = rnn.pack_padded_sequence(tanhed, q_len ,batch_first=True)
        output, (h,c) = self.lstm(packed)
        #c를 사용하는 이유: 우리는 질문 전체를 관통하는(cell state) 특징을 얻고 싶으므로
        #output 썼으면 다음에 나올거 같은 놈을 쓰는거고
        #h는 대체 언제쓰는거지
        #따라서 c는 *1024
        return c.squeeze(0)

#Stp 3. Attention & Classifier
class Attention(nn.Module):
    def __init__():
        super().__init__()
        self.input_feat = 4096 
        self.mid_feat = 512
        self.output_feat = 2
        self.conv1 = nn.Conv2d(self.input_feat,self.mid_feat,1)
        self.conv2 = nn.Conv2d(self.mid_feat, self.output_feat, 1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=0)
        self.fc1 = nn.Linear(3072,1024)
        self.fc2 = nn.Linear(1024,3000)
        self.drop = nn.Dropout(0.5)
    def forward(q_feat,v_feat):
        #1. We concatenate tiled LSTM state with image features over depth dimension
        #So, Tile lstm state(Question feature)
        #현재 q_feat: 1*batch_sie *1024
        """
        비교해볼 것:처음부터 v와 q의 out_feat수를 맞춰주면 어떻게 되나
        """
        #tiled_q_feat: tile q_feat 1*1024 -> 14*14*2048
        tiled_q_feat = q_feat.tile((14,14,2))
        #Concat
        att = torch.cat((v_feat,tiled_q_feat),2)
        #Pass Conv1 & ReLU
        att = self.relu(self.conv1(self.drop(att)))
        #Pass Conv2 & softmax
        att = self.conv2(self.drop(att))
        #Output: att = 14*14*2 
        #We use thiese distributinos to compute two image glimpses by computing the weighted average of image featuresw
        #1*1 batch 이유 : image와의 elementwise multiplication 위해서
        #n*n*(#feat)을 (n^2)*(#feat)으로 바꿔주기 -> 이쪽이 논문의 취지에 조금 더 부합(counts region by l)
        att = att.view(-1,2) #14*14*2 -> 196*2
        att = self.softmax(att)
        v_feat = v_feat.view(-1,2048) # image feature: 14*14*2048 -> 196*2048
        x = torch.empty(2,2048)
        for i in range(0,2):
            att_i = att[:,i].view(-1,1) # 196*1
            x_i = att_i * v_feat
            x_i = x_i.sum(dim=0)
            x[i,:] = x_i   #feature glimpse x => 2*2048
        
        #Concat image glimpses witht the state of LSTM
        cat = torch.cat((x, q_feat.tile(2,1)), -1)
        #Then pass through a fully connected layer of size 1024 with ReLU
        att = self.relu(self.fc1(self.drop(att)))
        #The ouptut is fed to a linear layer of size M = 3000 followed by softmax 
        result = self.softmax(self.fc2(self.drop(att)))
        return result # The output will be 2*3000 
    
#add 3 models(lstm,visual,attention) to 1 Model
class Net(nn.Module):
    def __init__(self, num_tokens):
        super(Net,self).__init__()
        self.lstm = LSTM(num_tokens)
        self.attention = Attention()
        #Inistialise weight
        for m in self.modules():
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(img,question,len_question):
        q_feat = self.lstm(question, list(len_question))
        v_feat = img
        result = self.attention(q_feat,v_feat)
        return result
