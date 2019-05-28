from fastai import *
from fastai.vision import *


PATH = "./data/"
sz=48
# point to data
# data = ImageClassifierData.from_paths(PATH,
#                 bs=4,
#                 tfms=tfms_from_model(resnet34, sz,
#                 aug_tfms=transforms_side_on, 
#                 max_zoom=1.1))
data = ImageDataBunch.from_folder('data', ds_tfms=get_transforms(), size=48)
#data.show_batch(rows=3,figsize=(6,6))

# choose model vgg16
#model = VGG16()

learn = cnn_learner(data,models.vgg16_bn,metrics=accuracy)
#learn = ConvLearner.pretrained(model,data,precompute=True)

learn.freeze()
# freeze layers up to the last one, so weights will not be updated.

learning_rate = [0.001,0.01,0.1]
learn.fit(5,learning_rate)
learn.lr_find()

#learn.recorder.plot()


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)
learn.export()

learn.save('vgg16_bn')

log_preds = learn.predict()

print("shape:", log_preds.shape)

'''
log_preds = learn.predict()
print("shape:", log_preds.shape)

preds = np.argmax(log_preds, axis=1)
probs = np.exp(log_preds[:, 1])


def rand_by_mask(mask): 
    return np.random.choice(np.where(mask)[0],
                            min(len(preds), 4), 
                            replace=False)


def rand_by_correct(is_correct): 
    return rand_by_mask((preds == data.val_y) 
                               == is_correct)


def plots(ims, figsize=(12, 6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


def load_img_id(ds, idx):
    return np.array(PIL.Image.open(PATH+ds.fnames[idx]))


def plot_val_with_title(idxs, title):
    imgs = [load_img_id(data.val_ds, x) for x in idxs]
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(imgs, rows=1, titles=title_probs, figsize=(16, 8)) if len(imgs) > 0 else print('Not Found.')


plot_val_with_title(rand_by_correct(True), "Correctly classified")

plot_val_with_title(rand_by_correct(False), "Incorrectly classified")


def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]


def most_by_correct(y, is_correct):
    mult = -1 if (y == 1) == is_correct else 1
    return most_by_mask(((preds == data.val_y) == is_correct) & (data.val_y == y), mult)


plot_val_with_title(most_by_correct(0, True), "Most correct cats")

plot_val_with_title(most_by_correct(1, True), "Most correct dogs")

plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")

plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")

most_uncertain = np.argsort(np.abs(probs - 0.5))[:4]
plot_val_with_title(most_uncertain, "Most uncertain predictions")

learn = ConvLearner.pretrained(arch, data, precompute=True)

lrf = learn.lr_find()

learn.sched.plot_lr()

learn.sched.plot()

tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


def get_augs():
    data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
    x, _ = next(iter(data.aug_dl))
    return data.trn_ds.denorm(x)[1]


ims = np.stack([get_augs() for i in range(6)])

plots(ims, rows=2)

arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True)

learn.fit(1e-2, 1)

learn.precompute = False

learn.fit(1e-2, 3, cycle_len=1)

learn.sched.plot_lr()

learn.save('224_lastlayer')

learn.load('224_lastlayer')

learn.unfreeze()

lr = np.array([1e-4, 1e-3, 1e-2])

learn.fit(lr, 3, cycle_len=1, cycle_mult=2)

learn.sched.plot_lr()

learn.save('224_all')

learn.load('224_all')

log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)

accuracy_np(probs, y)

preds = np.argmax(probs, axis=1)
probs = probs[:, 1]


cm = confusion_matrix(y, preds)

plot_confusion_matrix(cm, data.classes)


plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")

plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")

tfms = tfms_from_model(resnet34, sz)

data = ImageClassifierData.from_paths(PATH, tfms=tfms)

learn = ConvLearner.pretrained(resnet34, data, precompute=True)

learn.fit(1e-2, 1)


def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))


acts = np.array([1, 0, 0, 1])
preds = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts, preds)
'''
