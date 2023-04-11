from __init__ import*
from torchsummary import summary
import torch

# get data 
mytransform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((480,640)), #did you resize the mask 
        transforms.ConvertImageDtype(torch.float),
    ])

dataset_one=Coco_Stuff_things(mytransform)
#train_set, val_set = torch.utils.data.random_split(dataset_one, [dataset_one.num_samples - 10, 10])

#data loader 
train_loader=DataLoader(dataset=dataset_one,batch_size=1,shuffle=True)
#val_loader=DataLoader(dataset=val_set,batch_size=1,shuffle=True)

# model 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'the device is {DEVICE}')
model=Ustuff().to(device=DEVICE)


# freeze the backbone

'''
for child in model.children():
    for id,ch in enumerate(child.children()):
        if id==0:
            for param in ch.parameters():
                #param.requires_grad = False
                print(param.requires_grad )


print(model)

for child in model_ft.children()[0]:
    for param in child.parameters():
        param.requires_grad = False
'''

#criterion
#loss_fn=nn.BCELoss() #nn.CrossEntropyLoss()
loss_fn=nn.CrossEntropyLoss()
#optimizer 
optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #lr was 0.001

#train
epochs=20
train(epochs,train_loader,model,loss_fn,optimizer,DEVICE)

#
#summary(model, [(3,288,512), (3,288,512)])


#acc=inference(model_path=r'Classagnostic_Segmentation\saved_model\model.pth',val_loader=val_loader,DEVICE=DEVICE,draw=True)
#print(acc)

