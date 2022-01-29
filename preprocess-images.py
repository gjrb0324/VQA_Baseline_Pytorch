import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import torch.nn.init as init
import multiprocessing
import numpy as np
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms  as T


import config
import data
import utils


class Net(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        resnet152=models.resnet152(pretrained=True,progress=True)
        print('resnet loaded')
        #기존 resnet152의 마지막 fc layer 제외한 모든 layer 받아옴
        self.features = nn.ModuleList(resnet152.children())[:-2]
        #Sequential로 넣어줌
        self.features = nn.Sequential(*self.features)
        input_features = resnet152.fc.in_features #input features of previous fc layer
        """
        #필요하면 임의의 layer 추가 가능

        self.fc0 = nn.Linear(in_features, 256)
        self.fc-bn = nn.BatchNorm1d(256, eps = 1e-2)
        
        #Module들의 layer xavier로 초기화\n",
        for m in self.modules():
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                init.xavier_uniform_(m.weight)
        """        
    def forward(self, input_img):
        #'pi is three dimensional tensor from the last layer of the residual network(resnet152),14*14**2048'\n",
        output = self.features(input_img)
        #l2Normalization
        batch, feat, m, n = output.size() #output: [batch, feat, m, n]
        output = output.view(batch, feat, -1).transpose(2,1).view(batch,-1,feat) #output : [batch, m*n,feat]
        #output = output/(1e+28) #pull down values in the 1e+28 world to 1e+0
        output_img = output/torch.norm(output, 2, -1, keepdim=True)
        return output_img.view(batch,m,n,feat) #output img: batch * 14*14*2048


def create_coco_loader(*args): #args : contain information about paths and available_workers
    paths = args[:-2]
    available_workers = args[-2]
    transform = args[-1]
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers= available_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
        available_workers = torch.cuda.device_count()
    else:
        available_workers = multiprocessing.cpu_count()
    print(device + " is available")
    print("num of available workers {}".format(available_workers))
    net = nn.DataParallel(Net()).to(device)
    net.eval()

    transform = T.Compose([T.Resize((config.image_size,config.image_size)), T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    loader = create_coco_loader(config.train_path, config.val_path, available_workers, transform)

    features_shape = (
        len(loader.dataset),
        config.output_size,
        config.output_size,
        config.output_features,
    )

    with h5py.File(config.preprocessed_path, 'w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        with torch.no_grad():
            for ids, imgs in tqdm(loader):
                imgs = imgs.to(device)
                out = net(imgs)

                j = i + imgs.size(0)

                features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
                coco_ids[i:j] = ids.numpy().astype('int32')
                i = j


if __name__ == '__main__':
    main()
