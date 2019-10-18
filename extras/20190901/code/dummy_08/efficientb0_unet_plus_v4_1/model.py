#https://github.com/junfu1115/DANet

from common import *
from efficientnet import *


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]


'''
0 torch.Size([1, 32, 128, 128])
1 torch.Size([1, 16, 128, 128])*
2 torch.Size([1, 24, 64, 64])*
3 torch.Size([1, 40, 32, 32])*
4 torch.Size([1, 80, 16, 16])
5 torch.Size([1, 112, 16, 16])*
6 torch.Size([1, 192, 8, 8])
7 torch.Size([1, 320, 8, 8])*
8 torch.Size([1, 1280, 8, 8])*

'''


####################################################################################################
def upsize2(x):
    #x = F.interpolate(x, size=e.shape[2:], mode='nearest')
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d( out_channel//2),
            Swish(), #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel//2),
            Swish(), #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),

            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(out_channel),
            Swish(), #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.top(torch.cat(x, 1))
        return x



class Net(nn.Module):
    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(Net, self).__init__()

        self.mix = nn.Parameter(torch.FloatTensor(5))
        self.mix.data.fill_(1)

        self.e = EfficientNet(drop_connect_rate)
        self.block = [
            self.e.stem,
            self.e.block1, self.e.block2, self.e.block3, self.e.block4, self.e.block5, self.e.block6, self.e.block7,
            self.e.last
        ]
        self.e.logit = None  #dropped


        self.decode0_1 =  Decode(16+24, 24)

        self.decode1_1 =  Decode(24+40, 40)
        self.decode0_2 =  Decode(16+24+40, 40)

        self.decode2_1 =  Decode(40+112, 112)
        self.decode1_2 =  Decode(24+40+112, 112)
        self.decode0_3 =  Decode(16+24+40+112, 112)

        self.decode3_1 =  Decode(112+1280, 128)
        self.decode2_2 =  Decode(40+112+128, 128)
        self.decode1_3 =  Decode(24+40+112+128, 128)
        self.decode0_4 =  Decode(16+24+40+112+128, 128)

        self.logit1 = nn.Conv2d(24,num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(40,num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(112,num_class, kernel_size=1)
        self.logit4 = nn.Conv2d(128,num_class, kernel_size=1)


    def forward(self, x):
        batch_size,C,H,W = x.shape

        #----------------------------------
        #extract efficientnet feature

        backbone = []
        for i in range( len(self.block)):
            x = self.block[i](x)
            #print(i, x.shape)

            if i in [1,2,3,5,8]:
                backbone.append(x)


        #----------------------------------
        x0_0 = backbone[0] # 16
        x1_0 = backbone[1] # 24
        x0_1 = self.decode0_1([x0_0, upsize2(x1_0)])

        x2_0 = backbone[2] # 40
        x1_1 = self.decode1_1([x1_0, upsize2(x2_0)])
        x0_2 = self.decode0_2([x0_0, x0_1, upsize2(x1_1)])

        x3_0 = backbone[3] #112
        x2_1 = self.decode2_1([x2_0, upsize2(x3_0)])
        x1_2 = self.decode1_2([x1_0, x1_1, upsize2(x2_1)])
        x0_3 = self.decode0_3([x0_0, x0_1, x0_2, upsize2(x1_2)])


        x4_0 = backbone[4] #1280
        x3_1 = self.decode3_1([x3_0, upsize2(x4_0)])
        x2_2 = self.decode2_2([x2_0, x2_1, upsize2(x3_1)])
        x1_3 = self.decode1_3([x1_0, x1_1, x1_2, upsize2(x2_2)])
        x0_4 = self.decode0_4([x0_0, x0_1, x0_2, x0_3, upsize2(x1_3)])


        # deep supervision
        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)

        logit = self.mix[1]*logit1 + self.mix[2]*logit2 + self.mix[3]*logit3 + self.mix[4]*logit4
        logit = F.interpolate(logit, size=(H,W), mode='bilinear', align_corners=False)
        return logit  #logit, logit0


#########################################################################


def flat_dice_loss(logit, truth):
    smooth = 1.0

    probability = torch.sigmoid(logit)
    p = probability.view(-1)
    t = truth.view(-1)
    loss = ((2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth))
    return  loss

# -(1-p)**gamma * log(p)
def focal_loss(logit, truth, gamma=2):

    #https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    #https://discuss.pytorch.org/t/numerical-stability-of-bcewithlogitsloss/8246
    logit = logit.view(-1)
    truth = truth.view(-1)

    max  = (-logit).clamp(min=0)
    loss = logit - logit * truth  +  max + ((-max).exp() + (-logit - max).exp()).log()
    inv_prob = F.logsigmoid(-logit * (truth * 2.0 - 1.0))
    loss = (inv_prob * gamma).exp() * loss
    loss = loss.mean()
    return loss

def soft_dice_loss(logit, truth, weight=[1,8]):
#def soft_dice_loss(logit, truth, weight=[1,1]):

    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)
    w = t.detach()
    w = w*weight[1]+(1-w)*weight[0]

    p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice.mean()
    return loss

#---
#def criterion(logit, truth, weight=[0.75,0.25]):
def criterion(logit, truth, weight=None):
    logit = logit.view(-1)
    truth = truth.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    if weight is None:
        loss = loss.mean()

    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).sum()
        #raise NotImplementedError

    return loss


#----

def metric(logit, truth, threshold=0.5):
    batch_size,num_class,H,W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size*num_class,H*W)
        truth = truth.view(batch_size*num_class,H*W)

        probability = torch.sigmoid(logit)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum==0)
        pos_index = torch.nonzero(t_sum>=1)
        #print(len(neg_index), len(pos_index))

        dice_neg = (p_sum == 0).float()
        dice_pos = 2* (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice     = torch.cat([dice_pos,dice_neg])
        num_neg  = len(dice_neg)
        num_pos  = len(dice_pos)

        dice_neg = np.nan_to_num(dice_neg.mean().item(),0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(),0)

    return dice_pos,dice_neg, num_pos,num_neg



##############################################################################################
def make_dummy_data(folder='256x256', batch_size=8):

    image_file =  glob.glob('/root/share/project/kaggle/2019/steel/data/dump/%s/image/*.png'%folder) #32
    image_file = sorted(image_file)

    input=[]
    truth=[]
    for b in range(0, batch_size):
        i = b%len(image_file)
        image = cv2.imread(image_file[i], cv2.IMREAD_COLOR)
        mask  = np.load(image_file[i].replace('/image/','/mask/').replace('.png','.npy'))

        input.append(image)
        truth.append(mask)

    input = np.array(input)
    input = image_to_input(input)
    truth = np.array(truth)
    truth = (truth>0).astype(np.float32)

    return input, truth


#########################################################################
def image_to_input(image):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-IMAGE_RGB_MEAN[0])/IMAGE_RGB_STD[0]
    input[:,1] = (input[:,1]-IMAGE_RGB_MEAN[1])/IMAGE_RGB_STD[1]
    input[:,2] = (input[:,2]-IMAGE_RGB_MEAN[2])/IMAGE_RGB_STD[2]
    return input


def input_to_image(input):
    input = input.data.cpu().numpy()
    input[:,0] = (input[:,0]*IMAGE_RGB_STD[0]+IMAGE_RGB_MEAN[0])
    input[:,1] = (input[:,1]*IMAGE_RGB_STD[1]+IMAGE_RGB_MEAN[1])
    input[:,2] = (input[:,2]*IMAGE_RGB_STD[2]+IMAGE_RGB_MEAN[2])
    input = input.transpose(0,2,3,1)
    input = (input*255).astype(np.uint8)
    image = input[...,::-1]
    return image






#########################################################################
def run_check_efficientnet():
    net = Net()
    #print(net)
    load_pretrain(net.e)


def run_check_net():

    batch_size = 1
    C, H, W    = 3, 256, 256
    num_class  = 4

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net(num_class=num_class).cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ',input.shape)
    print('logit: ',logit.shape)
    #print(net)


def run_check_train():

    num_class = 4
    if 1:
        input, truth = make_dummy_data(batch_size=16)
        batch_size, C, H, W  = input.shape

        print(input.shape)
        print(truth.shape)


    #---
    truth = torch.from_numpy(truth).float().cuda()
    input = torch.from_numpy(input).float().cuda()


    net = Net(drop_connect_rate=0.1).cuda()
    load_pretrain(net.e, skip=['logit'],is_print=False)#

    net = net.eval()
    with torch.no_grad():
        logit = net(input)
        loss = criterion(logit, truth)

        dice_pos,dice_neg, num_pos,num_neg = metric(logit, truth)
        print('loss = %0.5f'%loss.item())
        print('dice_pos,dice_neg = %0.5f,%0.5f '%(dice_pos,dice_neg))
        print('num_pos,num_neg   = %d,%d '%(num_pos,num_neg))
        print('')



    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =',batch_size)
    print('-------------------------------------------------')
    print('[iter ]  loss     |  dice_pos,neg,    ')
    print('-------------------------------------------------')
          #[00000]  0.70383  | 0.00000, 0.46449


    i=0
    optimizer.zero_grad()
    while i<=200:
        #if i>100: weight=None

        net.train()
        optimizer.zero_grad()


        logit = net(input)
        loss = criterion(logit, truth)
        dice_pos,dice_neg, num_pos,num_neg = metric(logit, truth)


        (loss).backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f  | %0.5f,%0.5f '%(
                i,
                loss.item(),
                dice_pos,dice_neg
            ))
        i = i+1
    print('')


    if 1:
        print(net.mix)

        #net.eval()
        logit = net(input)
        probability = torch.sigmoid(logit)
        probability = probability.data.cpu().numpy()
        truth = truth.data.cpu().numpy()

        image = input_to_image(input)
        for b in range(batch_size):
            print('%d : '%(b))
            result = np.hstack([
                np.vstack([v for v in truth[b]]),
                np.vstack([p for p in probability[b]]),
            ])

            image_show_norm('result',result, resize=0.5)
            image_show('image',image[b], resize=0.5)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_efficientnet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


