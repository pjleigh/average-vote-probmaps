import torch
import numpy as np
from PIL import Image
import glob
from statistics import mean
import matplotlib.pyplot as plt

# Variables
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9] # list of float numbers 0-1 to threshold for majority voting or averaging
topknums = [3, 4, 5] # list of int numbers of top masks to take when deciding, 1-10
segmapDir = './segprobmaps/' # string directory of where all npy segmentation probability maps are for a single image
outDir = './out/' # directory to save len(topknums) * len(thresholds) plot of new masks based on majority voting or averaging
option = "majority" # string "majority" or "average" to pick between options

'''
gets (512x512x10) segmentation probability maps ranging (0-1), threshold value (0-1), and topknum (1-10)
outputs binary (0 or 1) segmentation mask (512x512) based on threshold and majority voting of topk 
''' 
def majorityvotemask(segmaps, threshold, topknum):
    n, m, _ = segmaps.shape
    segmask = np.zeros((n, m))
    topkpred = torch.topk(torch.tensor(segmaps), topknum, 2)
    
    for i in range(n):
        for j in range(m):
            # count number of predictions in topk above treshold
            binarylist = [int(x>=threshold) for x in topkpred.values[i, j, :]]
            
            # if majority of topk above threshold, set output segmask to 1 
            if ((sum(binarylist)/topknum) >= 1/2):
                segmask[i, j] = 1
    
    return segmask

'''
gets (512x512x10) segmentation probability maps ranging (0-1), threshold value (0-1), and topknum (1-10)
outputs binary (0 or 1) segmentation mask (512x512) based on threshold and averaging of topk 
'''
def averagemask(segmaps, threshold, topknum):
    n, m, _ = segmaps.shape
    segmask = np.zeros((n, m))

    topkpred = torch.topk(torch.tensor(segmaps), topknum, 2)
    
    for i in range(n):
        for j in range(m):
            # get values of predictions in topk
            values = [tensor.item() for tensor in topkpred.values[i, j, :]]
            
            # if average of topk above threshold, set output segmask to 1 
            if (mean(values) >= threshold):
                segmask[i, j] = 1
                    
    return segmask

def main():
    segmaplist = [f for f in glob.glob(segmapDir+"**.npy", recursive=False)]

    for l in range(len(segmaplist)):
        loadedsegmaps = np.load(segmaplist[l])[:, :, 1, :]
        k = 0
        segmask = np.zeros((512, 512, (len(topknums) * len(thresholds))))

        for i in range(len(topknums)):
            for j in range(len(thresholds)):

                if option == "majority":
                    segmask[:, :, k] = majorityvotemask(loadedsegmaps, thresholds[j], topknums[i])
                elif option == "average":
                    segmask[:, :, k] = averagemask(loadedsegmaps, thresholds[j], topknums[i])
                else:
                    print("invalid option. Change to \"majority\" or \"average\".")

                plt.subplot((len(topknums)),len(thresholds), k+1)
                plt.imshow(Image.fromarray((segmask[:, :, k] * 255).astype(np.uint8)))
                plt.axis('off')
                
                k = k + 1
                
        plt.savefig(outDir+str(segmaplist[l].split("/")[-1].split(".")[-2])+"_maskfig.png", dpi=200)

if __name__ == "__main__":
    main()