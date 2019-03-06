import torch
import  torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import models.crnn as crnn
import numpy as np
import os

model_path = '/home/lzhpersonal/crnn.pytorch/expr/checkpoint-adam.pth'
alphabet = '0123456789'

model = crnn.CRNN(32, 1, 11, 256)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def luhn(digit):
    factor = [2,1,2,1,2,1,2,1,2,1,2]
    code = [x%10+x//10 for x in [factor[i]*digit[i] for i in range(11)]]
    check = sum(code)%10
    if check != 0:
        check = 10 - check
    return check

def predict(image):
    image = resizeNormalize(image, (200, 32))
    if torch.cuda.is_available():
        image = image.to(device)
    image = image.view(1, *image.size())

    model.eval()
    preds = model(image)
    preds = preds.squeeze()
    preds = F.softmax(preds, 1)
    preds = preds.detach().cpu().numpy()
    probs = np.sort(preds, 1)[:, ::-1]
    preds = np.argsort(preds, 1)[:, ::-1] - 1

    t = preds[:,0]
    index = []
    label = []
    for i in range(t.shape[0]):
        if t[i] != -1 and (not (i > 0 and t[i - 1] == t[i])):
            index.append(i)
            label.append(preds[i, 0])

    if len(label) == 13:
        for i in range(len(index)):
            ind = np.argsort(probs[index, 0])[i]
            digit = [label[i] for i in range(13) if i != ind]
            if luhn(digit[:11])==digit[11]:
                del label[ind]
                break

    if len(label)==12 and luhn(label[:11])!=label[11]:
        probs = probs[index]
        preds = preds[index]
        probs_2 = [probs[i][1] if preds[i][1] != -1 else probs[i][2] for i in range(12)]
        preds_2 = [preds[i][1] if preds[i][1] != -1 else preds[i][2] for i in range(12)]

        for i in range(5):
            ind = np.argsort(probs_2)[::-1][i]
            digit = label.copy()
            digit[ind] = preds_2[ind]
            if luhn(digit[:11]) == digit[11]:
                label = digit
                break

    return label

def resizeNormalize(img, size):

    img = img.resize(size, Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)
    return img



if __name__ == '__main__':
    image = Image.open('/home/lzhpersonal/crnn.pytorch/crnn_images/1215_1203_002_right-a.jpg').convert('L')
    label = predict(image)
    print(label)

































