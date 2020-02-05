
# coding: utf-8

# In[1]:


from functions import *
import cmapy
from pytorchtools3 import EarlyStopping

torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code
time1 = time.time()
# print("RandomBrightnessContrast")
# modelNo
# Unet --> 0
# Deeplab --> 1
# HED --> 2
# RCF --> 3
# CED --> 4

# In[2]:


seismic_path = '/data/anyu/thebetraintest/seistrain.npy'
label_path = '/data/anyu/thebetraintest/faulttrain.npy'


# In[3]:


t_start = time.time()
seismic = np.load(seismic_path)
fault = np.load(label_path)
print("load in {} sec".format(time.time()-t_start))


# In[4]:


# print(seismic.shape, fault.shape)
# seismic = np.moveaxis(seismic[4:1807],-2,-1)
print(seismic.shape, fault.shape)
print(seismic.max(),seismic.min(), fault.max(), fault.min())
# # reorder input data to same order IL, Z, XL


# # In[5]:


# seismic = (seismic-seismic.min(axis=(1,2), keepdims=True))/(seismic.max(axis=(1,2), keepdims=True)-seismic.min(axis=(1,2), keepdims=True))
# print(seismic.shape)


# # In[6]:


IL, Z, XL = fault.shape


# In[7]:


best_model_fpath = 'hed_96_48_trainaug99_900200_trainvalstep100_seed_thresholdGT.model'
best_iou_threshold=0.5
epoches = 100
patience = 20
im_height = Z
im_width = XL
splitsize = 96
stepsize = 48
overlapsize = splitsize-stepsize
pixelThre = int(0.03*splitsize*splitsize)
print(pixelThre)

# In[8]:


modelNo = 2
if modelNo == 0:
    from model_zoo.UNET import Unet
    model = Unet()
    print("use model Unet")
elif modelNo == 1:
    from model_zoo.DEEPLAB.deeplab import DeepLab
    model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
    print("use model DeepLab")
elif modelNo == 2:
    from model_zoo.HED import HED
    model = HED()
    print("use model HED")
elif modelNo == 21:
    from model_zoo.HED_crop import HED
    model = HED(sizeaftercrop=stepsize)
    print("use model HED")
elif modelNo == 3:
    from model_zoo.RCF import RCF
    model = RCF()
    print("use model RCF")
else:
    from model_zoo.CED import CED
    model = CED()
    print("use model CED")
model.cuda();
summary(model, (1, splitsize, splitsize))


horizontal_splits_number = int(np.ceil((im_width-overlapsize)/stepsize))
print("horizontal_splits_number", horizontal_splits_number)
width_after_pad = stepsize*horizontal_splits_number+overlapsize
print("width_after_pad", width_after_pad)
left_pad = int((width_after_pad-im_width)/2)
right_pad = width_after_pad-im_width-left_pad
print("left_pad,right_pad",left_pad,right_pad)

vertical_splits_number = int(np.ceil((im_height-overlapsize)/stepsize))
print("vertical_splits_number",vertical_splits_number)
height_after_pad = stepsize*vertical_splits_number+overlapsize
print("height_after_pad",height_after_pad)
top_pad = int((height_after_pad-im_height)/2)
bottom_pad = height_after_pad-im_height-top_pad
print("top_pad,bottom_pad", top_pad,bottom_pad)


# In[17]:


t_start = time.time()
X = []
Y = []
for i in range(0,900,100):
    mask = fault[i]
    splits = split_Image(mask, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
#     print(splits.shape)
    t = (splits.sum((1,2)) < pixelThre)
    no_label_element_index = list(compress(range(len(t)), t))
    # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
    splits = np.delete(splits, no_label_element_index,0) # delete element i along axis 0
#     print("splits.shape", splits.shape)
    Y.extend(splits)
    
    img = seismic[i]
    splits = split_Image(img, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
    splits = np.delete(splits, no_label_element_index,0) # delete element i along axis 0
#     print("splits.shape", splits.shape)
    X.extend(splits)
#     break

print(len(Y))
print(len(X))
print(X[0].shape)
print("read images in {} sec".format(time.time()-t_start))


# In[20]:


t_start = time.time()
X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
print(X.shape)
print(Y.shape)
print("read images in {} sec".format(time.time()-t_start))


# In[22]:


if len(Y.shape) == 3:
    Y = np.expand_dims(Y, axis=-1)
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)
print(X.shape)
print(Y.shape)

X_train = X
Y_train = Y


t_start = time.time()
X = []
Y = []
for i in range(900,1100,100):
    mask = fault[i]
    splits = split_Image(mask, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
#     print(splits.shape)
    t = (splits.sum((1,2)) < pixelThre)
    no_label_element_index = list(compress(range(len(t)), t))
    # get all the indexes of the no label pieces by adding elements in axis 2 and 3.
    splits = np.delete(splits, no_label_element_index,0) # delete element i along axis 0
#     print("splits.shape", splits.shape)
    Y.extend(splits)
    
    img = seismic[i]
    splits = split_Image(img, True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number)
    splits = np.delete(splits, no_label_element_index,0) # delete element i along axis 0
#     print("splits.shape", splits.shape)
    X.extend(splits)
#     break

print(len(Y))
print(len(X))
print(X[0].shape)
print("read images in {} sec".format(time.time()-t_start))


# In[20]:


t_start = time.time()
X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)
print(X.shape)
print(Y.shape)
print("read images in {} sec".format(time.time()-t_start))


# In[22]:


if len(Y.shape) == 3:
    Y = np.expand_dims(Y, axis=-1)
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)
print(X.shape)
print(Y.shape)

X_val = X
Y_val = Y

print("X_train",X_train.shape)
print("X_val",X_val.shape)

print("Y_train",Y_train.shape)
print("Y_val",Y_val.shape)


# In[ ]:

aug_times = 99

t_start = time.time()
origin_train_size = len(X_train)
print(origin_train_size)
X_train_aug = np.zeros((origin_train_size*aug_times,splitsize,splitsize,1))
Y_train_aug = np.zeros((origin_train_size*aug_times,splitsize,splitsize,1))
for i in range(len(X_train)):
    for j in range(aug_times):
        aug = strong_aug(p=1)
        augmented = aug(image=X_train[i], mask=Y_train[i])
        X_train_aug[origin_train_size*j + i] = augmented['image']
        Y_train_aug[origin_train_size*j + i] = augmented['mask']
print("read images in {} sec".format(time.time()-t_start))

X_train_aug = X_train_aug.astype(np.float32)
Y_train_aug = Y_train_aug.astype(np.float32)
if len(X_train)==origin_train_size:
    X_train = np.append(X_train,X_train_aug, axis=0)
if len(Y_train)==origin_train_size:
    Y_train = np.append(Y_train, Y_train_aug, axis=0)
print("X_train after aug",X_train.shape) 
print("Y_train after aug",Y_train.shape)
print("read images in {} sec".format(time.time()-t_start))
X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_train = Y_train>=0.5
Y_train = Y_train.astype(np.float32)
#-----------------------
# t_start = time.time()
# origin_val_size = len(X_val)
# print(origin_val_size)
# X_val_aug = np.zeros((origin_val_size*aug_times,splitsize,splitsize,1))
# Y_val_aug = np.zeros((origin_val_size*aug_times,splitsize,splitsize,1))
# for i in range(len(X_val)):
#     for j in range(aug_times):
#         aug = strong_aug(p=1)
#         augmented = aug(image=X_train[i], mask=Y_train[i])
#         X_val_aug[origin_val_size*j + i] = augmented['image']
#         Y_val_aug[origin_val_size*j + i] = augmented['mask']
# print("read images in {} sec".format(time.time()-t_start))

# X_val_aug = X_val_aug.astype(np.float32)
# Y_val_aug = Y_val_aug.astype(np.float32)
# if len(X_val)==origin_val_size:
#     X_val = np.append(X_val,X_val_aug, axis=0)
# if len(Y_val)==origin_val_size:
#     Y_val = np.append(Y_val, Y_val_aug, axis=0)
# print("X_val after aug",X_val.shape) 
# print("Y_val after aug",Y_val.shape)
# print("read images in {} sec".format(time.time()-t_start))
# X_val = X_val.astype(np.float32)
# Y_val = Y_val.astype(np.float32)
#-----------------------
X_train = np.moveaxis(X_train,-1,1)
Y_train = np.moveaxis(Y_train,-1,1)
X_val = np.moveaxis(X_val,-1,1)
Y_val = np.moveaxis(Y_val,-1,1)
print("X_train",X_train.shape)
print("X_val",X_val.shape)
print("Y_train",Y_train.shape)
print("Y_val",Y_val.shape)


# In[ ]:


# idea from: https://www.kaggle.com/erikistre/pytorch-basic-u-net
class faultsDataset(torch.utils.data.Dataset):

    def __init__(self,preprocessed_images,train=True, preprocessed_masks=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.images = preprocessed_images
        self.masks = preprocessed_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
#         mask = None
# #         if self.train:
        mask = self.masks[idx]
        return (image, mask)


# In[32]:


faults_dataset_train = faultsDataset(X_train, train=True, preprocessed_masks=Y_train)
faults_dataset_val = faultsDataset(X_val, train=False, preprocessed_masks=Y_val)
# faults_dataset_test = faultsDataset(X_test, train=False, preprocessed_masks=Y_test)

batch_size = 64 

train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                           batch_size=batch_size, 
                                           shuffle=False)

# test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test, 
#                                            batch_size=batch_size, 
#                                            shuffle=False)


# In[ ]:



# criterion = nn.BCEWithLogitsLoss()
# learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)
print("optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)")
if modelNo == 0 or modelNo == 1:
    print("optimizer = torch.optim.Adam(model.parameters(), lr=0.01)")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
elif modelNo == 4:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=0.0002)
    print("optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=0.0002)")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5, verbose=True)


# In[ ]:


bceloss = nn.BCELoss()
mean_train_losses = []
mean_val_losses = []
mean_train_accuracies = []
mean_val_accuracies = []
t_start = time.time()
early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0)
for epoch in range(epoches):                  
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    labelled_val_accuracies = []

    model.train()
    for images, masks in train_loader: 
        torch.cuda.empty_cache()
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        if modelNo == 21:
            masks = crop(masks,stepsize,stepsize)
        outputs = model(images)
        
        loss = torch.zeros(1).cuda()
        y_preds = outputs
        if modelNo == 0 or modelNo == 1 or modelNo == 4:
#             print("bceloss")
            loss = bceloss(outputs, masks)
#             loss = cross_entropy_loss_HED(outputs, masks)
#             loss = nn.BCEWithLogitsLoss(outputs, masks) 
        elif modelNo == 2 or modelNo == 21:
            for o in range(5):
                loss = loss + cross_entropy_loss_HED(outputs[o], masks)
            loss = loss + bceloss(outputs[-1],masks)
            y_preds = outputs[-1]
        elif modelNo == 3:
            for o in outputs:
                loss = loss + cross_entropy_loss_RCF(o, masks)
            y_preds = outputs[-1]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), masks.squeeze(1).byte())
        train_accuracies.append(train_acc.mean())        

    model.eval()
    for images, masks in val_loader:
        torch.cuda.empty_cache()
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        if modelNo == 21:
            masks = crop(masks,stepsize,stepsize) 
        outputs = model(images)
        
        loss = torch.zeros(1).cuda()
        y_preds = outputs
        if modelNo == 0 or modelNo == 1 or modelNo == 4:
#             print("bceloss")
            loss = bceloss(outputs, masks)
#             loss = cross_entropy_loss_HED(outputs, masks)
#             loss = nn.BCEWithLogitsLoss(outputs, masks) 
        elif modelNo == 2 or modelNo == 21:
            for o in range(5):
                loss = loss + cross_entropy_loss_HED(outputs[o], masks)
            loss = loss + bceloss(outputs[-1],masks)
            y_preds = outputs[-1]#(outputs[0]+outputs[1]+outputs[2]+outputs[3]+outputs[4]+outputs[5])/len(outputs)
        elif modelNo == 3:
            for o in outputs:
                loss = loss + cross_entropy_loss_RCF(o, masks)
            y_preds = outputs[-1]
        val_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte())
        val_accuracies.append(val_acc.mean())

#         todelete = torch.sum(masks,dim=(2,3))<1
#         no_label_element_index = list(compress(range(len(todelete)), todelete))
# #         print(no_label_element_index)
#         labelled_val_acc = np.delete(val_acc, no_label_element_index,0) 
#         labelled_val_accuracies.extend(labelled_val_acc)
        
    mean_train_losses.append(torch.mean(torch.stack(train_losses)))
    mean_val_losses.append(torch.mean(torch.stack(val_losses)))
    mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
    mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))
    
    scheduler.step(torch.mean(torch.stack(val_losses)))    
    early_stopping(torch.mean(torch.stack(val_losses)), model, best_model_fpath)
    
#     mean_train_losses.append(np.mean(train_losses))
#     mean_val_losses.append(np.mean(val_losses))
#     mean_train_accuracies.append(np.mean(train_accuracies))
#     mean_val_accuracies.append(np.mean(val_accuracies))
#     scheduler.step(np.mean(val_losses))   
#     early_stopping(np.mean(val_losses), model, best_model_fpath)

    if early_stopping.early_stop:
        print("Early stopping")
        break
        
#     # load the last checkpoint with the best model
#     model.load_state_dict(torch.load('checkpoint.pt'))
    torch.cuda.empty_cache()
    
    for param_group in optimizer.param_groups:
        learningRate = param_group['lr']
    
    
    # Print Epoch results
    t_end = time.time()
#     print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
#           .format(epoch+1, np.mean(train_losses), np.mean(val_losses), torch.mean(torch.stack(train_accuracies)).item(), torch.mean(torch.stack(val_accuracies)).item(), t_end-t_start, learningRate))
#     t_start = time.time()
#     print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Labelled Val IoU: {}. Time: {}. LR: {}'
#           .format(epoch+1, np.mean(train_losses), np.mean(val_losses),np.mean(train_accuracies), np.mean(val_accuracies), np.mean(labelled_val_accuracies), t_end-t_start, learningRate))
    print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
          .format(epoch+1, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(train_accuracies)), torch.mean(torch.stack(val_accuracies)), t_end-t_start, learningRate))
    
    t_start = time.time()
    
#     torch.save(model.state_dict(), best_model_fpath)


# In[ ]:
mean_train_losses = np.asarray(torch.stack(mean_train_losses).cpu())
mean_val_losses = np.asarray(torch.stack(mean_val_losses).cpu())
mean_train_accuracies = np.asarray(torch.stack(mean_train_accuracies).cpu())
mean_val_accuracies = np.asarray(torch.stack(mean_val_accuracies).cpu())

fig = plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
train_loss_series = pd.Series(mean_train_losses)
val_loss_series = pd.Series(mean_val_losses)
train_loss_series.plot(label="train_loss")
val_loss_series.plot(label="validation_loss")
plt.legend()
plt.subplot(1, 2, 2)
train_acc_series = pd.Series(mean_train_accuracies)
val_acc_series = pd.Series(mean_val_accuracies)
train_acc_series.plot(label="train_acc")
val_acc_series.plot(label="validation_acc")
plt.legend()
plt.savefig('{}_loss_acc.png'.format(best_model_fpath))

totaltime = time.time()-time1
print("total cost {} hours".format(totaltime/3600))
