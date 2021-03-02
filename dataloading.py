import torch
import math
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PokeDataset(Dataset):
	def __init__(self):
		runs = sorted(glob.glob("poke/train/*"))
		self.images = []
		self.actions = []

		for itter in tqdm(range(1)):
			run = runs[itter]
			actions = np.load(run+"/actions.npy")
			self.px_max = max(actions[:,0])
			self.px_min = min(actions[:,0])
			self.py_max = max(actions[:,1])
			self.py_min = min(actions[:,1])
			for i in range(len(actions)):
				if(actions[i][4] == 1):
					img_before = plt.imread(run + "/img_%04d.jpg"%i)
					img_after = plt.imread(run + "/img_%04d.jpg"%(i+1))
					self.images.append([img_before,img_after])
					px = self.loc(actions[i][0], self.px_max, self.px_min)
					py = self.loc(actions[i][1], self.py_max, self.py_min)
					t = self.angle(actions[i][2])
					l = self.length(actions[i][3])
					self.actions.append([px,py,t,l])

	def __len__(self):
		return(len(self.actions))

	def __getitem__(self, index):
		return((self.images[index],self.actions[index]))

	def loc(self, p, maxVal, minVal):
		x = np.zeros((20))
		if(p == maxVal):
			x[19] = 1
			return(x)
		x[int((p-minVal)//((maxVal-minVal)/20))] = 1
		return(x)

	def angle(self, theta):
		x = np.zeros((36))
		x[int(theta//(math.pi/13))] = 1
		return(x)

	def length(self, l):
		x = np.zeros((11))
		x[int((l-.01)//12)] = 1
		return(x)


data = PokeDataset()
dataLodaer = DataLoader(data, batch_size=5, shuffle=True)
print(next(iter(dataLoader)))