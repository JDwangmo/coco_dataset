#!encoding=utf8

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import sys

dataDir='..'
# dataType='train2014'
dataType='val2014'

if dataType =='test2014':
    annFile = '%s/annotations/image_info_%s.json'%(dataDir,dataType)
else:
    annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
    
# initialize COCO api for instance annotations
coco=COCO(annFile)

catIds = coco.getCatIds(catNms=['person']);
imgIds = coco.getImgIds(catIds=catIds );
print '类别为人的图片数量为：%d'%len(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# print img


# 是否显示图片和标注
show_image = False

# 遍历包含人的图片
for image_count, image_id in enumerate(imgIds):
    sys.stderr.write('\r第%d张图片： image id: %d'%(image_count,image_id))
    img = coco.loadImgs(image_id)[0]
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    
    I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
#     io.imsave(
#         '%s/images_processed/images/%s'%(dataDir,img['file_name']),
#         I
#     )
    if show_image:
        plt.figure(); plt.axis('off')
        plt.imshow(I)
        coco.showAnns(anns)
# 所有物体的masks，初始为全0    
    masks = np.zeros(I.shape[:2])
    for ann_count,ann in enumerate(anns):
        mask = coco.annToMask(ann)
        masks += mask
# 0-1图片
    masks = masks>0
# 绘制0-1 mask
    if show_image:
        plt.figure(); plt.axis('off')
        plt.imshow(masks,cmap=plt.cm.gray)
    np.savetxt(open('%s/images_processed/masks/%s.txt'%(dataDir,img['file_name']),'w'),masks,fmt='%d',delimiter=',')