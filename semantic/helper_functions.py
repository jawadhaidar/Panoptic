from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def train(epochs,train_loader,model,loss_fn,optimizer,DEVICE):

    for epoch in tqdm(range(epochs)):
        running_loss=0
        for i, batched_sample in enumerate(train_loader):
        
            img,mask=batched_sample #mask shape will be bxmxn with numbers in range [0,nclasses-1] 
            #forward pass
            out=model(img.to(device=DEVICE)).float()  # the out shape will be bxmxnxk where k is number of stuff classes
            #calculate lossd  type=torch.long
            loss = loss_fn(out, mask.squeeze(1).long().to(device=DEVICE)) 
            #print(f'max of out {out.max()} and maxof label {label.max()}')
            #plt.imshow(out[0,0,:,:].squeeze().detach().numpy())
            #plt.show()
            #backward pass
            loss.backward()
            #update weights
            optimizer.step()
            #zero gradients
            optimizer.zero_grad()
            #print batch number out of total 
            #print epoch number
            # Gather data and report
            running_loss += loss.item()
            print(f'batch {i} out of {len(train_loader)} with batch loss {loss.item():.20f} and acc : {naive_accuracy(out,mask)}')

        print(f'epoch {epoch} || loss : {running_loss}')
        torch.save(model, r'Panoptic\semantic\models\model.pth') #to be changed

        

def inference(model_path,val_loader,DEVICE,draw):
    model=torch.load(model_path).to(device=DEVICE)
    acc_sum=0
    model.eval()
    for i, batched_sample in enumerate(val_loader):
        img,mask=batched_sample
        img=img.to(device=Device)
        mask=mask.to(device=Device)

        #forward passss
        out=model(img).float()
        
        acc_batch=naive_accuracy(out,label)
        print(acc_batch)
        #draw 
        if draw:
            for ind in range(out.shape[0]): # for every sample in batch
                a=out[ind,0,:,:].squeeze().detach().to('cpu').numpy()
                plt.subplot(1,2,1)
                plt.title("predicted")
                plt.imshow(a)
                plt.subplot(1,2,2)
                plt.title("ground truth")
                plt.imshow(mask[ind,0,:,:].squeeze().detach().to('cpu').numpy()) #fix this 
                #plt.savefig(os.path.join(f'Classagnostic_Segmentation\results\validation_withdilmask',"{i}.png"))
                plt.show()

        acc_sum+=acc_batch

    acc_total=acc_sum/ i+1
    return acc_total

def naive_accuracy(predicted_mask,gt_mask):
    #change the predicted mask logits into probabilities
    softmax=nn.Softmax(dim=1)
    predicted_mask=softmax(predicted_mask) #along the depth dim 
    #do argmax on the depth channel 
    predicted_mask=torch.argmax(predicted_mask,dim=1) 
    gt_mask=gt_mask.squeeze(1)
    #compare to matrices
    TPs=predicted_mask==gt_mask
    
    batch_size=predicted_mask.shape[0]
    total=(predicted_mask.shape[-1]*predicted_mask.shape[-2]) * batch_size
    print(f'TPs: {TPs.sum()},total: {total}')
    acc= TPs.sum()/total * 100
    return acc



if __name__=="__main__":
    predicted=torch.ones((100,3,10,10))
    groundtruth=torch.zeros((100,1,10,10))

    acc=naive_accuracy(predicted_mask=predicted, gt_mask=groundtruth)
    print(acc)
    
