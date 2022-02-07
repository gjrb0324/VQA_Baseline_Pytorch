import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
import os
import config
import data
import model
import utils


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_iterations = 0

def run(net, device, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    #loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))
    
    if train:
        #loss = 0.0
        for v, q, a, idx, q_len in tq:
            v = v.to(device)
            q = q.to(device)
            a = a.to(device)
            q_len = q_len.to(device)
            
            optimizer.zero_grad()

            #try new loss fucntion to avoid NaN
            out = net(v, q, q_len)

            #transform a(one hot) to not one-hot
            a = a.argmax(dim=-1) #a : a[batch]= answer_index
 
            mini_batch_loss = 0.0
            for i in range(out.size(0)): #for each elements in batch,
                mini_batch_loss += -torch.log(out[i][a[i]]) #nll : negative log likelihood
                if mini_batch_loss.isnan()== True:
                    print('Weight of net : {}\n'.format(net.state_dict()))
                    sys.exit('\n nan loss detected while calculating mini_batch_loss')

            mini_batch_loss /= float(out.size(0))
            if mini_batch_loss.isnan() == True:
                print('Weight of net : {}\n'.format(net.state_dict()))
                sys.exit('\n nan loss detected after calculating mini_batch_loss')

            global total_iterations
            #loss = mini_batch_loss + loss * total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            mini_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()

            total_iterations += 1
            #loss = loss / total_iterations            

            fmt = '{:.4f}'.format
            tq.set_postfix(mini_batch_loss=fmt(mini_batch_loss)) #total_loss=fmt(loss))
            writer.add_scalar("Loss/train", mini_batch_loss, total_iterations-1)

    else:
        eval_iterations = epoch*1890 + 1
        with torch.no_grad():
            for v,q,a,idx, q_len in tq:
                v = v.to(device)
                q = q.to(device)
                a = a.to(device)
                q_len = q_len.to(device)

                out = net(v,q,q_len)
                acc = utils.batch_accuracy(out.data, a.data).cpu()
            # store information about evaluation of this minibatch
                _, answer = out.data.cpu().max(dim=1)
                answ.append(answer.view(-1))
                accs.append(acc.view(-1))
                idxs.append(idx.view(-1).clone())

                for a in acc:
                    acc_tracker.append(a.item())
                fmt = '{:.4f}'.format
                tq.set_postfix(acc=fmt(acc_tracker.mean.value))
                writer.add_scalar("Accuracy/eval", acc_tracker.mean.value, eval_iterations)
                eval_iterations+=1

            answ = list(torch.cat(answ, dim=0))
            accs = list(torch.cat(accs, dim=0))
            idxs = list(torch.cat(idxs, dim=0))
            print('\nanswer: {}, predicted {}, idx {}'.format(a,out,idx))
            return answ, accs, idxs


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
        available_workers = torch.cuda.device_count()
    else :
        available_workers = len(os.sched_getaffinity(0))
    print(device + " is available")
    print('num of workers {}'.format(available_workers))
    
    #Open files for writting loss and accuracy
    
    
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('/data/VQA_Baseline_Pytorch/logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(available_workers,train=True)
    val_loader = data.get_loader(available_workers, val=True)

    net = nn.DataParallel(model.Net(device,train_loader.dataset.num_tokens)).to(device)
    print('Net loaded: Success')
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(net,device, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = run(net,device, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)
        '''
        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'vocab': train_loader.dataset.vocab,
        }
        torch.save(results, target_name)
        '''
    writer.close()

if __name__ == '__main__':
    main()
