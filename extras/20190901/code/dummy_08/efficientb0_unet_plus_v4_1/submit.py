import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *
from train   import *

 
def flip_lr_augment(image, mask, infor):
    image, mask = do_flip_lr(image, mask)
    return image, mask, infor

def flip_lr_inverse_augment(predict, truth):
    predict = predict[:,:,:,::-1]
    truth   = truth[:,:,:,::-1]
    return predict, truth

#---

def flip_ud_augment(image, mask, infor):
    image, mask = do_flip_ud(image, mask)
    return image, mask, infor

def flip_ud_inverse_augment(predict, truth):
    predict = predict[:,:,::-1,:]
    truth   = truth[:,:,::-1,:]
    return predict, truth

#---

def null_augment(image, mask, infor):
    return image, mask, infor

def null_inverse_augment(predict, truth):
    return predict, truth



######################################################################################

def post_process(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict





def compute_metric(image_id, truth, predict):

    num = len(image_id)
    t = truth.reshape(num*4,-1).astype(np.float32)
    p = predict.reshape(num*4,-1).astype(np.float32)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    d_neg = (p_sum == 0).astype(np.float32)
    d_pos = 2* (p*t).sum(-1)/((p+t).sum(-1)+1e-12)

    t_sum = t_sum.reshape(num,4)
    p_sum = p_sum.reshape(num,4)
    d_neg = d_neg.reshape(num,4)
    d_pos = d_pos.reshape(num,4)

    #for each class
    dice_neg = []
    dice_pos = []
    for c in range(4):
        neg_index = np.where(t_sum[:,c]==0)[0]
        pos_index = np.where(t_sum[:,c]>=1)[0]
        dice_neg.append(d_neg[:,c][neg_index])
        dice_pos.append(d_pos[:,c][pos_index])

    ##
    dice_neg_all = np.concatenate(dice_neg).mean()
    dice_pos_all = np.concatenate(dice_pos).mean()
    dice_neg = [np.nan_to_num(d.mean(),0) for d in dice_neg]
    dice_pos = [np.nan_to_num(d.mean(),0) for d in dice_pos]

    ## from kaggle probing ...
    kaggle_pos = np.array([ 128,43,741,120 ])
    kaggle_neg_all = 6172
    kaggle_all     = 1801*4
    #
    dice_all = (dice_neg_all*kaggle_neg_all + sum(dice_pos*kaggle_pos))/kaggle_all
    return dice_all, dice_neg_all, dice_pos_all, dice_neg, dice_pos





###################################################################################


def do_evaluate(net, test_dataset, test_augment, test_inverse_augment):

    test_dataset.augment = test_augment

    test_loader = DataLoader(
            test_dataset,
            sampler     = SequentialSampler(test_dataset),
            #sampler     = RandomSampler(test_dataset),
            #sampler     = FixedSampler(test_dataset, list(np.arange(0,20))),
            batch_size  = 1,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate
    )

    test_num  = 0
    test_id   = []
    test_image = []
    test_probability = []
    test_truth = []

    start = timer()
    for b, (input, truth, infor) in enumerate(test_loader):

        with torch.no_grad():
            net.eval()
            input = input.cuda()
            truth = truth.cuda()

            logit =  data_parallel(net,input)  #net(input)
            probability = torch.sigmoid(logit)
            loss = criterion(logit, truth)
            dice_pos,dice_neg, num_pos,num_neg = metric(logit, truth)

        #---
        batch_size = len(infor)
        test_id.extend([i.image_id for i in infor])
        test_probability.append(probability.data.cpu().numpy())
        test_truth.append(truth.data.cpu().numpy())
        test_image.append(input_to_image(input))
        test_num += batch_size

        #---
        print('\r %4d / %4d  %s'%(
            test_num, len(test_loader.dataset), time_to_str((timer() - start),'min')
        ),end='',flush=True)

   # assert(test_num == len(test_loader.dataset))
    print('')

    test_image = np.concatenate(test_image)
    test_probability = np.concatenate(test_probability)
    test_truth = np.concatenate(test_truth)
    test_probability, test_truth = test_inverse_augment(test_probability, test_truth)

    return test_id, test_image, test_probability, test_truth





#################################################################################################################
def run_submit():

    out_dir = \
        '/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1'

    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1/checkpoint/00090000_model.pth'
         #'/root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1/checkpoint/00072000_model.pth'


    mode    = 'test' #'train' # 'test'
    augment = ['null','flip_lr','flip_ud'] #'null' #'flip_lr'  #


    ## setup  -----------------------------------------------------------------------------

    os.makedirs(out_dir +'/submit/%s/%s'%(mode,augment), exist_ok=True)
    os.makedirs(out_dir +'/submit/%s/%s/dump'%(mode,augment), exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset -------

    log.write('** dataset setting **\n')
    if mode == 'train':
        test_dataset = SteelDataset(
            mode    = 'train',
            csv     = ['train.csv',],
            split   = ['valid0_500.npy',],
            augment = None,
        )

    if mode == 'test':
        test_dataset = SteelDataset(
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_1801.npy',],
            augment = None, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)

    ## net ----------------------------------------
    log.write('** net setting **\n')

    net = Net().cuda()
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n'%(type(net)))
    log.write('\n')

    ## start testing here! ##############################################
    if 1:
        image_id, image, probability, truth = None, None, None, None
        for k, a in enumerate(augment):
            test_augment, test_inverse_augment = {
                'null': (null_augment, null_inverse_augment),
                'flip_lr': (flip_lr_augment, flip_lr_inverse_augment),
                'flip_ud': (flip_ud_augment, flip_ud_inverse_augment),
            }[a]


            log.write('** augment = %s **\n'%(a))
            i, m, p, t = do_evaluate(net, test_dataset, test_augment, test_inverse_augment)
            if k==0:
                image_id = i
                image = m
                probability = p
                truth = t
            else:
                if k==1: probability = probability**0.5
                probability += p**0.5


            # write_list_to_file(out_dir +'/submit/%s/%s/image_id.txt'%(mode,augment),image_id)
            # np.save(out_dir +'/submit/%s/%s/image.npy'%(mode,augment),image)
            # np.save(out_dir +'/submit/%s/%s/probability.npy'%(mode,augment),probability)
            # np.save(out_dir +'/submit/%s/%s/truth.npy'%(mode,augment),truth)

        probability = probability/len(augment)
        print(len(image_id))
        print(image.shape)
        print(probability.shape)
        print(truth.shape)

        #exit (0)

    #---
    # image_id = read_list_from_file(out_dir +'/submit/%s/%s/image_id.txt'%(mode,augment))
    # image = np.load(out_dir +'/submit/%s/%s/image.npy'%(mode,augment))
    # probability = np.load(out_dir +'/submit/%s/%s/probability.npy'%(mode,augment))
    # truth = np.load(out_dir +'/submit/%s/%s/truth.npy'%(mode,augment))

    num_test = len(image_id)

    ##debug
    if 0:
        for b in range(num_test):
            print(b)
            overlay = draw_predict_result(
                image[b], truth[b], probability[b],
                scale=1, stack='vertical'
            )

            image_show('overlay',overlay,0.5)
            cv2.imwrite(out_dir +'/submit/%s/%s/dump/%s.png'%(mode,augment,image_id[:-4]),overlay)

            #cv2.imwrite(out_dir +'/train/%05d.png'%(di*100+b), overlay)
            cv2.waitKey(0)



    ################################################################################
    '''
   
    '''

    #threshold_pixel = 0.5
    #min_size=3500
    threshold_pixel = [0.5,0.5,0.6,0.5]
    min_size = [800,1000,3000,3500]


    predict1 = (probability > np.array(threshold_pixel).reshape(1,4,1,1))
    predict2 = predict1.copy()
    for b in range(num_test):
        print('\r post_process: b=%d/%d'%(b,num_test-1), end='')
        for c in range(4):
            predict2[b,c] = post_process(predict2[b,c], min_size[c])


    print('')
    print(predict2.shape)
    print('')

    if mode =='train':
        truth   = (truth > 0.5)

        dice_all, dice_neg_all, dice_pos_all, dice_neg, dice_pos = compute_metric(image_id, truth, predict1)

        log.write('\n')
        log.write('augment = %s\n'%augment)
        log.write('threshold_pixel = %s\n'%str(threshold_pixel))
        log.write('dice_all     = %f\n'%dice_all)
        log.write('dice_neg_all = %f\n'%dice_neg_all)
        log.write('dice_pos_all = %f\n'%dice_pos_all)
        for c in range(4):
            log.write('dice_neg[%d], dice_pos[%d], = %0.5f,  %0.5f\n'%(
                c+1,c+1,dice_neg[c],dice_pos[c]
            ))
        log.write('\n')

        #---

        dice_all, dice_neg_all, dice_pos_all, dice_neg, dice_pos = compute_metric(image_id, truth, predict2)
        log.write('min_size = %s\n'%str(min_size))
        log.write('dice_all     = %f\n'%dice_all)
        log.write('dice_neg_all = %f\n'%dice_neg_all)
        log.write('dice_pos_all = %f\n'%dice_pos_all)
        for c in range(4):
            log.write('dice_neg[%d], dice_pos[%d], = %0.5f,  %0.5f\n'%(
                c+1,c+1,dice_neg[c],dice_pos[c]
            ))
        log.write('\n')

    if mode =='test':
        csv_file = out_dir +'/submit/%s/%s/submission.csv'%(mode,augment)

        image_id_class_id = []
        encoded_pixel = []

        for b in range(num_test):
            for c in range(4):
                image_id_class_id.append(image_id[b]+'_%d'%(c+1))

                rle = run_length_encode(predict2[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)




'''

--- [START 2019-09-01_20-57-28] ----------------------------------------------------------------

	@common.py:  
	set random seed
		SEED = 1567342649
	set cuda environment
		torch.__version__              = 1.1.0
		torch.version.cuda             = 9.0.176
		torch.backends.cudnn.version() = 7501
		os['CUDA_VISIBLE_DEVICES']     = 0
		torch.cuda.device_count()      = 1



	SEED         = 1567342649
	PROJECT_PATH = /root/share/project/kaggle/2019/steel/code/dummy_08
	__file__     = /root/share/project/kaggle/2019/steel/code/dummy_08/efficientb0_unet_plus_v4_1/submit.py
	out_dir      = /root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1

** dataset setting **
test_dataset : 
	mode    = train
	split   = ['valid0_500.npy']
	csv     = ['train.csv']
		len   =   500
		num   =  2000
		neg   =  1712  0.856
		pos   =   288  0.144
		pos1  =    35  0.070  0.122
		pos2  =     5  0.010  0.017
		pos3  =   213  0.426  0.740
		pos4  =    35  0.070  0.122


** net setting **
	initial_checkpoint = /root/share/project/kaggle/2019/steel/result/efficientb0_unet_plus_v4_1/checkpoint/00090000_model.pth
<class 'model.Net'>

** augment = null **
  500 /  500   0 hr 01 min
** augment = flip_lr **
  500 /  500   0 hr 01 min
** augment = flip_ud **
  500 /  500   0 hr 01 min
500
(500, 256, 1600, 3)
(500, 4, 256, 1600)
(500, 4, 256, 1600)
 post_process: b=499/499
(500, 4, 256, 1600)


augment = ['null', 'flip_lr', 'flip_ud']
threshold_pixel = [0.5, 0.5, 0.6, 0.5]
dice_all     = 0.865270
dice_neg_all = 0.907126
dice_pos_all = 0.625015
dice_neg[1], dice_pos[1], = 0.83871,  0.44403
dice_neg[2], dice_pos[2], = 0.96970,  0.26193
dice_neg[3], dice_pos[3], = 0.79791,  0.66653
dice_neg[4], dice_pos[4], = 0.97634,  0.60525

min_size = [800, 1000, 3000, 3500]
dice_all     = 0.914999
dice_neg_all = 0.985397
dice_pos_all = 0.502669
dice_neg[1], dice_pos[1], = 0.98065,  0.36040
dice_neg[2], dice_pos[2], = 0.99798,  0.19072
dice_neg[3], dice_pos[3], = 0.94774,  0.52413
dice_neg[4], dice_pos[4], = 1.00000,  0.55889
'''




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()



