# paths
qa_path = 'C:\\coco_questoins'  # directory containing the question and annotation jsons
train_path = 'C:\\Images\\train2014'  # directory of training images
val_path = 'C:\\Images\\val2014'  # directory of validation images
test_path = 'C:\\Images\\test2015'  # directory of test images
vocabulary_path = 'Z:\\gyu\\VQA_Baseline_Pytorch\\vocab.json'  # path where the used vocabularies for question and answers are saved to
preprocessed_path = 'Z:\\gyu\\\VQA_Baseline_Pytorch\\preprocessed.h5'

task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 2
max_answers = 3000
batch_size = 128
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
