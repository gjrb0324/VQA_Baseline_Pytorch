import torch
#import torchtext
#from torchtext import data as torchtextdata
#from torchtext.data import get_tokenizer
#from torchtext.data import Iterator
import torch.nn as nn
import config
import model
import json
import data
#from torchtext.data import TabularDataset

def find_question(x):
    return x['question']

def recover_seq(sequence, mode= None, q_len = None):
    with open(config.vocabulary_path, 'r') as fd:
        vocab_json = json.load(fd)
    if mode == 0:
        vocab = vocab_json['question']
        key_list = list(vocab.keys())
        result = [[key_list[word-1] for word in question] for question\
                  in sequence]
    if mode == 1:
        vocab= vocab_json['answer']
        key_list = list(vocab.keys())
        seq = [torch.argmax(answers) for answers in sequence]
        result = [key_list[answer] for answer in seq]

    return result

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device + "is available")
    '''
    tokenizer= get_tokenizer("basic_english")
    Text = data.Field(sequential = True, use_vocab= True, \
                               tokenize=tokenizer,lower= True, \
                               batch_first = True)

    with open("../coco_questions/OpenEnded_mscoco_train2014_questions.json", \
              'r') as f:
        questions = json.load(f)

    mapped = list(map(find_question, test_data['questions']))
    with open('pre_vocab.txt', 'w') as f:
        [f.write(question+ '\n') for question in mapped ]

    test_data = TabularDataset.splits(path='./', test='pre_vocab.txt',\
                                      format='tsv',fields=[('text',TEXT)])[0]


    TEXT.build_vocab(test_data)

    test_loader = Iterator(dataset=test_data, batch_size = 4,shuffle=False, \
                           sort= False,sort_within_batch=False)

    #End of question dataloader

    with open("../coco_questions/mscoco_train_2014_annotations.json", 'r') as f:
        answer_json = json.load(f)

    answers = [[a['answer'] for a in answ_dict['answers']] for ans_dict in \
               answers_json['annotations']]

    Answer = data.Field(use_vocab= True, batch_first = True)
    data.Dataset(answers, [('answer'), Answer)])

    Answer.build_vocab(answers)


    dataiter = iter(test_loader
    )
    '''
    val_loader = data.get_loader(0,val=True)
    net = nn.DataParallel(model.Net(device, val_loader.dataset.num_tokens)).to\
        (device)
    net.load_state_dict(torch.load('trained_net.pth'))

    with torch.no_grad():
        net.eval()
        for v,q,a,idx,q_len in val_loader:
            v = v.to(device)
            q = q.to(device)
            a = a.to(device)
            q_len = q_len.to(device)
            out,att = net(v,q,q_len)
            v = v + att
            print('Questions \n')
            for question in recover_seq(q,0,q_len):
                print(question)
                print('\n')
            print('Answers \n')
            print(recover_seq(a,1))
            print('Predicted Answers \n')
            print(recover_seq(out,1))
            print('Focused Image \n')
            print(v)





if __name__ == '__main__':
    main()
