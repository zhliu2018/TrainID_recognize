from libdarknet import performDetect
from PIL import Image
import numpy as np
from crnn import predict, luhn
import os
import time
import matplotlib.pyplot as plt

def recognize(image):

    detections  = performDetect(image=image, showImage=False)

    if len(detections)==0:return None
    detections = np.array([d[2] for d in detections])
    detections[:,0] -= detections[:,2]/2
    detections[:,1] -= detections[:,3]/2
    #detections = detections[detections[:,1].argsort()]

    detections[:, 2] += detections[:,0]
    detections[:, 3] += detections[:,1]
    trainID = []


    if len(detections) == 1:
        trainID.append(detections)
    else:
        detections = detections[detections[:, 0].argsort()]
        x_coord = detections[:, 0]
        start = 0
        for i in range(len(x_coord)-1):
            if x_coord[i+1] - x_coord[i] > 50:
                trainID.append(detections[start:i+1, :])
                start = i+1
                break
        trainID.append(detections[start:, :])

    label = None
    for i in trainID:
        if i.shape[0] == 3 or i.shape[0] == 4:
            i = i[i[:,1].argsort()]
            if i.shape[0]==4:
                i = i[:3,:]
            img1 = image.crop(i[0])
            w1, h1 = img1.size
            img1 = img1.resize((int(32 * w1 / h1), 32))
            img2 = image.crop(i[1])
            w2, h2 = img2.size
            img2 = img2.resize((int(32 * w2 / h2), 32))
            img3 = image.crop(i[2])
            w3, h3 = img3.size
            img3 = img3.resize((int(32 * w3 / h3), 32))
            img = Image.new('RGB', (int(w1 / h1 * 32) + int(w2 / h2 * 32) + int(w3 / h3 * 32), 32))
            img.paste(img1, (0, 0))
            img.paste(img2, (int(w1 / h1 * 32), 0))
            img.paste(img3, (int(w2 / h2 * 32) + int(w1 / h1 * 32), 0))
            img = img.convert('L')
            # img.save('0.jpg')
            label = predict(img)
            if len(label) < 12:
                i[2][0] -= 15
                i[2][2] += 15
                img3 = image.crop(i[2])
                w3, h3 = img3.size
                img3 = img3.resize((int(32 * w3 / h3), 32))
                img = Image.new('RGB', (int(w1 / h1 * 32) + int(w2 / h2 * 32) + int(w3 / h3 * 32), 32))
                img.paste(img1, (0, 0))
                img.paste(img2, (int(w1 / h1 * 32), 0))
                img.paste(img3, (int(w2 / h2 * 32) + int(w1 / h1 * 32), 0))
                img = img.convert('L')
                label = predict(img)
            if len(label)==12 and luhn(label[:11])==label[11]:
                break

        elif i.shape[0] == 1:
            img = image.crop(i[0])
            w, h = img.size
            img = img.resize((int(32 * w / h), 32))
            img = img.convert('L')
            # img.save('1.jpg')
            label = predict(img)
            if len(label)==12 and luhn(label[:11])==label[11]:
                break
            if len(label) < 12:
                i[0][0] -= 15
                i[0][2] +=15
                img = image.crop(i[0])
                w, h = img.size
                img = img.resize((int(32 * w / h), 32))
                img = img.convert('L')
                # img.save('1_enlarge.jpg')
                label = predict(img)
                if len(label) == 12 and luhn(label[:11]) == label[11]:
                    break
    return label


since = time.time()
logfile = open('trainID_recognize_log.txt', 'w')
correct = 0
num = 0
for line in open('test.txt'):
    imageName, label = line.strip().split()
    imagePath = os.path.join('./image', imageName)
    if not os.path.exists(imagePath):
        continue
    image = Image.open(imagePath)
    w, h = image.size
    right = 2400
    left = 0
    margin = 200
    result = None
    while right < w:
        img = image.crop((left, 600, right, 1800))
        left += 2400 - margin
        right += 2400 - margin
        result = recognize(img)
        if result is not None and len(result)>11:
            break
    if result is  None or len(result)<12:
        img = image.crop((w-2400, 600, w, 1800))
        result = recognize(img)

    if result != None:
        result = ''.join(str(i) for i in result)
    num += 1
    if result == label:
        correct += 1
        logfile.writelines([imageName, ' ', label, '\n'])
        print(imageName, ' ', label)
    elif result == None:
        logfile.writelines([imageName, ' ', label, ' ', 'fail','\n'])
        print(imageName, ' ', label, ' ', 'fail')
    else:
        logfile.writelines([imageName, ' ', label, ' ', result, '\n'])
        print(imageName, ' ', label, ' ', result)

        accuracy = correct / num
end = time.time()
logfile.write('correct pictures:{}, the accuracy is : {:.4f}'.format(correct, accuracy))
print('correct pictures:{}/{}, the accuracy is : {:.4}'.format(correct, num, accuracy))

print('total time:{} , average time: {}', end-since, (end-since)/num)

