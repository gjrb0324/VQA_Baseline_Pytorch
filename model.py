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
        self.lstm = nn.LSTM(300,1024,batch_first=True,dropout=0.5)
        #Embedding weight 초기화
        init.xavier_uniform_(self.embedding.weight)
        
        #위의 CNN에서와 같이 가중치를 xavier로, bias를 0으로 초기화
        #lstm.weight_ih_l0 : 0번째(1번쨰) layer의 input-hidden의 가중치 벡터 -> W_(ii,if,ig,io)
        #lstm.weight_hh_l0 : 0번째(1번째) layer의 hidden-hidden의 가중치 벡터 - > W_(hi,hf,hg,ho)
        #W_(i,f,g,o) : 각각 input, forget, cell, cell, output gate의 가중치들
        #원래 [[i_1],[i_2]...[i_n]],[[1f_1],[f_2]...[f_n]],[[g_1]..]..[..[o_n]],
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
        tanhed = self.tanh(embedded)
        packed = rnn.pack_padded_sequence(tanhed, q_len ,batch_first=True)
        output, (h,c) = self.lstm(packed)
        #c를 사용하는 이유: 우리는 질문 전체를 관통하는(cell state) 특징을 얻고 싶으므로
        #output 썼으면 다음에 나올거 같은 놈을 쓰는거고
        #h는 대체 언제쓰는거지
        #따라서 c는 *1024
        return h.squeeze(0)

#Stp 3. Attention & Classifier
class Attention(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.input_feat = 4096 
        self.mid_feat = 512
        self.output_feat = 2
        self.conv1 = nn.Conv2d(self.input_feat,self.mid_feat,1)
        self.conv2 = nn.Conv2d(self.mid_feat, self.output_feat, 1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(5120,1024)
        self.fc2 = nn.Linear(1024,3000)
        self.drop = nn.Dropout(0.5)	
    def forward(self,q_feat,v_feat):
        #1. We concatenate tiled LSTM state with image features over depth dimension
        #So, Tile lstm state(Question feature)
        #현재 q_feat: 1*batch_sie *1024
        #현재 v_feat: batch_size * 14* 14* 2048
        """
        비교해볼 것:처음부터 v와 q의 out_feat수를 맞춰주면 어떻게 되나
        """
        #tiled_q_feat: tile q_feat 1*1024 -> 14*14*2048
        b, n, m, f = v_feat.size()
        att = torch.empty(b,n,m,f*2).to(self.device)
        for i in range(b):
            q_feat_i = q_feat[i,:]
            v_feat_i = v_feat[i,:,:,:] #v_feat_i : 14*14*2048
            q_feat_i = q_feat_i.repeat(14,14,2) #q_feat_i: 1*1024 -> 14*14*2048
            att[i,:,:,:] = torch.cat((v_feat_i,q_feat_i),-1)
            
        #att : batc* 14*14*4096 -> batch*4096 * 14*14
        att = att.transpose(3,1)
        #Concat
        #Pass Conv1 & ReLU
        att = self.relu(self.conv1(self.drop(att)))
        #Pass Conv2 & softmax
        att = self.conv2(self.drop(att))
        #Output: att = 2*14*14 
        #We use thiese distributinos to compute two image glimpses by computing the weighted average of image featuresw
        #1*1 batch 이유 : image와의 elementwise multiplication 위해서
        #n*n*(#feat)을 (n^2)*(#feat)으로 바꿔주기 -> 이쪽이 논문의 취지에 조금 더 부합(counts region by l)
        b,f,n,m = att.size() #att = batch*feature *n *m
        att = att.view(b,f,-1) #batch*2*14*14 -> batch*2*196
        att = self.softmax(att) #softmax over spatial dimension
        #Weighted sum of attention distribution and v_feat
        v_feat = v_feat.view(b,-1,2048) # image feature: batch*14*14*2048 -> batch*196*2048
        x1 = att[:,0,:].view(b,-1,1) * v_feat.view(b,-1,2048)
        x2 = att[:,1,:].view(b,-1,1) * v_feat.view(b,-1,2048)
        x1 = x1.sum(dim=-2) #sum over spatial dimension    
        x2 = x2.sum(dim=-2)
        
        #x1, x2 now : attentino value with the shape batch*2048
        con_feat = torch.cat((x1,x2, q_feat), dim=-1) 
        con_feat = self.relu(self.fc1(self.drop(con_feat)))
        con_feat = self.softmax(self.fc2(self.drop(con_feat)))

        return con_feat # The output will be softmaxed batch*3000 
    
#add 3 models(lstm,visual,attention) to 1 Model
class Net(nn.Module):
    def __init__(self, device, num_tokens):
        super(Net,self).__init__()
        self.device = device
        self.lstm = LSTM(num_tokens)
        self.attention = Attention(self.device)
        #Inistialise weight
        for m in self.modules():
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self,img,question,len_question):
        q_feat = self.lstm(question, list(len_question)).to(self.device)
        v_feat = img.to(self.device)
        result = self.attention(q_feat,v_feat).to(self.device)
        return result
