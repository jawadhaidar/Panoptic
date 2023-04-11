import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


#print(modeledited)


class Ustuff(nn.Module):
    def __init__(self):
        super(Ustuff,self).__init__()
        self.num_maps=183 #in this case it is the number of stuff classes + things
        self.in_channels=3
        self.model1 = smp.Unet(
                encoder_name="resnext101_32x8d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.num_maps,                      # model output channels (number of classes in your dataset)
            ) 

        #freeze the backbone param
        for id,child in enumerate(self.model1.children()):
            if id==0:
                for param in child.parameters():
                    param.requires_grad=False
                    #print(param.requires_grad)


    def forward(self,x):
        out=self.model1(x)
        return out


if __name__=="__main__":

    mymodel=Ustuff()
    inp=torch.rand(1,3,512,512)
    print(mymodel(inp).shape)