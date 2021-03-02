import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Forward(nn.Module):
	def __init__(self):
		super(Forward, self).__init__()
		#Forward Model 
		#CNN
		self.conv1 = nn.Conv2d(3,96,11,4)
		self.maxPool1 = nn.MaxPool2d(3,2)
		self.conv2 = nn.Conv2d(96,256,5,1,2)
		self.maxPool2 = nn.MaxPool2d(3,2)
		self.conv3 = nn.Conv2d(256,384,3,1,1)
		self.conv4 = nn.Conv2d(384,384,3,1,1) 	
		self.conv5 = nn.Conv2d(384,256,3,1,1)
		self.maxPool3 = nn.MaxPool2d(3,2)
		#Length Prediction
		self.l1 = nn.Linear(18432,9000)
		self.l2 = nn.Linear(9000,1)
		#Angle Prediction
		self.theta1 = nn.Linear(18433,9000)
		self.theta2 = nn.Linear(9000,1)
		#Coordinate Prediction
		self.p1 = nn.Linear(18433,9000)
		self.p2 = nn.Linear(9000,2)

		#Inverse Model
		self.f1 = nn.Linear(9220,9220)
		self.f2 = nn.Linear(9220,9216)

	def latentRepresentation(self,x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.maxPool1(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.maxPool2(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = F.relu(x)
		x = self.conv5(x)
		x = F.relu(x)
		x = self.maxPool3(x)
		x = torch.flatten(x,1)
		return(x)

	def actionPrediction(self,x):
		l = self.l1(x)
		l = F.relu(l)
		l = self.l2(l)
		l = F.relu(l)
		res = torch.cat((l,x),1)
		theta = self.theta1(res)
		theta = F.relu(theta)
		theta = self.theta2(theta)
		theta = F.relu(theta)
		res = torch.cat((theta,x),1)
		p = self.p1(res)
		p = F.relu(p)
		p = self.p2(p)
		p = F.relu(p)
		return(l,theta,p)

	def inverse(self, action, x_0):
		cat = torch.cat((action,x_0),1)
		x = self.f1(cat)
		x = F.relu(x)
		x = self.f2(x)
		x = F.relu(x)
		return(x)

	def forward(self,I_0,I_1,action):
		x_0 = self.latentRepresentation(I_0)
		x_1 = self.latentRepresentation(I_1)
		concat = torch.cat((x_0,x_1),1)
		l,theta,p = self.actionPrediction(concat)
		x_1Bar = self.inverse(action, x_1)
		return(l,theta,p,x_1,x_1Bar)

def lossFunction(lBar,thetaBar,pBar,action,x_1,x_1Bar):
	ceLoss = nn.MSELoss()
	l1Loss = nn.L1Loss()
	return(ceLoss(action[:,:2],pBar)+
		   ceLoss(action[:,2:3],thetaBar)+
		   ceLoss(action[:,3:4],lBar)+
		   0.1*l1Loss(x_1,x_1Bar))



actionNumpy = torch.from_numpy(np.load("poke/train/run_00/actions.npy"))
action = actionNumpy[:2][:,:4].float()

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder("poke/train", transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

images, labels = next(iter(dataloader))
print(images.shape)
print(action)

forward = Forward()

lBar,thetaBar,pBar,x_1,x_1Bar = forward(images,images,action)
print(action[:,3:4],lBar)
print(lossFunction(lBar,thetaBar,pBar,action,x_1,x_1Bar))