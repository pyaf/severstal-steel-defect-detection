import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *



def valid_augment(image, mask, infor):
    return image, mask, infor



def train_augment(image, mask, infor):
    image, mask = do_random_crop(image, mask, 256, 256)
    #image = do_random_log_contast(image)

    if np.random.rand()>0.5:
        image, mask = do_flip_lr(image, mask)
    if np.random.rand()>0.5:
        image, mask = do_flip_ud(image, mask)

    # if np.random.rand()>0.5:
    #     image, mask = do_noise(image, mask)

    return image, mask, infor

def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        infor.append(batch[b][2])

    input = np.stack(input)
    input = image_to_input(input)

    truth = np.stack(truth)
    truth = (truth>0.5).astype(np.float32)

    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(truth).float()

    return input, truth, infor

#------------------------------------

def do_valid(net, valid_loader):

    valid_num  = np.zeros(3, np.float32)
    valid_loss = np.zeros(3, np.float32)

    for b, (input, truth, infor) in enumerate(valid_loader):

        #if b==5: break
        net.eval()
        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit =  data_parallel(net,input)  #net(input)
            loss = criterion(logit, truth)
            dice_pos,dice_neg, num_pos,num_neg = metric(logit, truth)

            #zz=0
        #---
        batch_size = len(infor)
        l = np.array([ loss.item(), dice_pos,dice_neg ])
        n = np.array([ batch_size, num_pos,num_neg ])
        valid_loss += l*n
        valid_num  += n

        #print(valid_loss)
        print('\r %8d /%8d'%(valid_num[0], len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss/valid_num

    return valid_loss



def run_train():

    out_dir = \
        '/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1'

    initial_checkpoint = \
        '/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1/checkpoint/00072000_model.pth'


    schduler = NullScheduler(lr=0.001)
    batch_size =  8 #8
    iter_accum =  2


    ## setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train',  exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['train0_12068.npy',],
        augment = train_augment,
    )
    train_loader  = DataLoader(
        train_dataset,
        #sampler     = BalanceClassSampler(train_dataset, 3*len(train_dataset)),
        sampler    = SequentialSampler(train_dataset),
        #sampler    = RandomSampler(train_dataset),
        batch_size  = batch_size,
        drop_last   = True,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    valid_dataset = SteelDataset(
        mode    = 'train',
        csv     = ['train.csv',],
        split   = ['valid0_500.npy',],
        augment = valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        #sampler    = SequentialSampler(valid_dataset),
        sampler     = RandomSampler(valid_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 8,
        pin_memory  = True,
        collate_fn  = null_collate
    )


    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage),strict=True)
    else:
        load_pretrain(net.e, skip=['logit'], is_print=False)


    log.write('%s\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    #net.set_mode('train',is_freeze_bn=True)
    #-----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    num_iters   = 3000*1000
    iter_smooth = 50
    iter_log    = 500
    iter_valid  = 1500
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 1500))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass



    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('                      |-------- VALID-------|---- TRAIN/BATCH ------------------\n')
    log.write('rate     iter   epoch |  loss  dice_pos,neg |  loss  dice_pos,neg | time        \n')
    log.write('--------------------------------------------------------------------------------\n')
              #0.00000    0.0*   0.0 |  0.674   0.09,0.54  |  0.000   0.00,0.00  |  0 hr 00 min

    train_loss = np.zeros(20,np.float32)
    valid_loss = np.zeros(20,np.float32)
    batch_loss = np.zeros(20,np.float32)
    iter = 0
    i    = 0


    start = timer()
    while  iter<num_iters:
        sum_train_loss = np.zeros(20,np.float32)
        sum = np.zeros(20,np.float32)

        optimizer.zero_grad()
        for input, truth, infor in train_loader:

            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch


            #if 0:
            if (iter % iter_valid==0):
                valid_loss = do_valid(net, valid_loader) #
                #pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                asterisk = '*' if iter in iter_save else ' '
                log.write('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f,%4.2f  |  %5.3f   %4.2f,%4.2f  | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:3],
                         *train_loss[:3],
                         time_to_str((timer() - start),'min'))
                )
                log.write('\n')


            #if 0:
            if iter in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                pass




            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth = truth.cuda()

            logit =  data_parallel(net,input)  #net(input)
            loss = criterion(logit, truth)
            dice_pos,dice_neg, num_pos,num_neg = metric(logit, truth)


            ((loss)/iter_accum).backward()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            l = np.array([ loss.item(), dice_pos,dice_neg ])
            n = np.array([ batch_size, num_pos,num_neg ])

            batch_loss[:3] = l
            sum_train_loss[:3] += l*n
            sum[:3] += n
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss/(sum+1e-12)
                sum_train_loss[...] = 0
                sum[...]            = 0


            print('\r',end='',flush=True)
            asterisk = ' '
            print('%0.5f  %5.1f%s %5.1f |  %5.3f   %4.2f,%4.2f  |  %5.3f   %4.2f,%4.2f  | %s' % (\
                         rate, iter/1000, asterisk, epoch,
                         *valid_loss[:3],
                         *batch_loss[:3],
                         time_to_str((timer() - start),'min'))
            , end='',flush=True)
            i=i+1



            # debug-----------------------------
            if 1:
                for di in range(3):
                    if (iter+di)%1000==0:

                        probability = torch.sigmoid(logit)
                        image = input_to_image(input)
                        truth = truth.data.cpu().numpy()
                        probability = probability.data.cpu().numpy()


                        for b in range(batch_size):
                            overlay = draw_predict_result(
                                image[b], truth[b], probability[b],
                                scale=1, stack='horizontal'
                            )

                            image_show('overlay',overlay,0.25)
                            cv2.imwrite(out_dir +'/train/%05d.png'%(di*100+b), overlay)
                            cv2.waitKey(1)
                            pass




        pass  #-- end of one data loader --
    pass #-- end of all iterations --

    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()



#####
#
#  03002.png
#
#  12
# 111111111111