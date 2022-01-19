import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
from tqdm import tqdm

import config
import data
import utils
from resnet import resnet as caffe_resnet


class Net(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        resnet152=models.resnet152(pretrained=True,progress=True)
        #기존 resnet152의 마지막 fc layer 제외한 모든 layer 받아옴
        self.features = nn.ModuleList(resnet152.children())[:-1]
        #Sequential로 넣어줌
        self.features = nn.Sequential(*self.features)
        input_features = resnet152.fc.in_features #input features of previous fc layer
        """
        #필요하면 임의의 layer 추가 가능

        self.fc0 = nn.Linear(in_features, 256)
        self.fc-bn = nn.BatchNorm1d(256, eps = 1e-2)
        """
        #Module들의 layer xavier로 초기화\n",
        for m in self.modules():
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                init.xavier_uniform_(m.weight)
                
    def forward(self, input_img):
        #'pi is three dimensional tensor from the last layer of the residual network(resnet152),14*14**2048'\n",
        output = self.features(input_img)
        print(output.size())
        #l2Normalization 
        output_img = input_img/torch.linalg.norm(output, 2, 1, keepdim=True)
        return output_img


def create_coco_loader(*paths):
    transform = utils.get_transform(config.image_size, config.central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.preprocess_batch_size,
        num_workers=config.data_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    print(device + " is available")

    net = Net().to(device)
    net.eval()

    loader = create_coco_loader(config.train_path, config.val_path)
    features_shape = (
        len(loader.dataset),
        config.output_features,
        config.output_size,
        config.output_size
    )

    with h5py.File(config.preprocessed_path, 'w', libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float16')
        coco_ids = fd.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        for ids, imgs in tqdm(loader):
            imgs = Variable(imgs.to(device), volatile=True)
            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
            coco_ids[i:j] = ids.numpy().astype('int32')
            i = j


if __name__ == '__main__':
    main()