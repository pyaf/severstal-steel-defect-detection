from common  import *


# https://www.kaggle.com/iafoss/severstal-fast-ai-256x256-crops-sub
# https://www.kaggle.com/rishabhiitbhu/unet-starter-kernel-pytorch-lb-0-88


def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != '':
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(' ')]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1  #???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask


def run_length_encode(mask):
    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        start  = np.where(m[1: ] > m[:-1])[0]+2
        end    = np.where(m[:-1] > m[1: ])[0]+2
        length = end-start

        rle = [start[0],length[0]]
        for i in range(1,len(length)):
            rle.extend([start[i],length[i]])

        rle = ' '.join([str(r) for r in rle])
    return rle


### draw ###################################################################

def mask_to_inner_contour(mask):
    mask = mask>0.5
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_contour_overlay(image, mask, color=(0,0,255), thickness=1):
    contour =  mask_to_inner_contour(mask)
    if thickness==1:
        image[contour] = color
    else:
        for y,x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x,y), thickness, color, lineType=cv2.LINE_4 )
    return image

def draw_mask_overlay(image, mask, color=(0,0,255), alpha=0.5):
    H,W,C = image.shape
    mask = (mask*alpha).reshape(H,W,1)
    overlay = image.astype(np.float32)
    overlay = np.maximum( overlay, mask*color )
    overlay = np.clip(overlay,0,255)
    overlay = overlay.astype(np.uint8)
    return overlay

def draw_predict_result(image, truth, probability, scale=1, stack='horizontal'):
    H,W,C = image.shape

    result = []
    image  = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    color = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
    for c in range(4):
        r = np.zeros((H,W,3),np.uint8)
        t = cv2.resize(truth[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        p = cv2.resize(probability[c], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        #r = draw_mask_overlay(r, t, (255,255,255), alpha=0.5)
        r = draw_mask_overlay(r, p, color[c], alpha=1)
        r = draw_contour_overlay(r, t, (255,255,255), thickness=2)

        image = draw_contour_overlay(image, t, color[c], thickness=2)
        result.append(r)

    result = [image,] + result
    if stack=='horizontal':
        result = np.hstack(result)
    if stack=='vertical':
        result = np.vstack(result)

    return result




### check ##############################################################

def run_check_rle():

    image = cv2.imread('/root/share/project/kaggle/2019/steel/data/train_images/002fc4e19.jpg',cv2.IMREAD_COLOR)
    value = [
        '002fc4e19.jpg_1','146021 3 146275 10 146529 40 146783 46 147038 52 147292 59 147546 65 147800 70 148055 71 148311 72 148566 73 148822 74 149077 75 149333 76 149588 77 149844 78 150100 78 150357 75 150614 72 150870 70 151127 67 151384 64 151641 59 151897 53 152154 46 152411 22',
        '002fc4e19.jpg_2','145658 7 145901 20 146144 33 146386 47 146629 60 146872 73 147115 86 147364 93 147620 93 147876 93 148132 93 148388 93 148644 93 148900 93 149156 93 149412 93 149668 46',
        '002fc4e19.jpg_3', '',
        '002fc4e19.jpg_4', '',
    ]
    rle = [value[i] for i in range(1,8,2)]


    mask = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])
    print(mask.shape)

    rle1 = [ run_length_encode(m) for m in mask ]
    print('0',rle1[0])
    print('1',rle1[1])
    print('2',rle1[2])
    print('3',rle1[3])
    assert(rle1==rle)
    print('check ok!!!!')
    exit(0)

    image_show_norm('mask[0]',mask[0],0,1)
    image_show_norm('mask[1]',mask[1],0,1)
    image_show_norm('mask[2]',mask[2],0,1)
    image_show_norm('mask[3]',mask[3],0,1)
    image_show('image',image)

    #---
    mask0 = draw_mask_overlay(image, mask[0],color=(0,0,255))
    image_show('mask0',mask0)
    mask1 = draw_mask_overlay(image, mask[1],color=(0,0,255))
    image_show('mask1',mask1)

    cv2.waitKey(0)


def run_make_split():

    image_file =  glob.glob('/root/share/project/kaggle/2019/steel/data/train_images/*.jpg')
    image_file = ['train_images/'+i.split('/')[-1] for i in image_file]
    print(len(image_file))
    print(image_file[:10])

    random.shuffle(image_file)
    print(image_file[:10])

    #12568
    num_valid = 500
    num_all   = len(image_file)
    num_train = num_all-num_valid

    train=np.array(image_file[num_valid:])
    valid=np.array(image_file[:num_valid])

    raise NotImplementedError
    np.save('/root/share/project/kaggle/2019/steel/data/split/train0_%d.npy'%len(train),train)
    np.save('/root/share/project/kaggle/2019/steel/data/split/valid0_%d.npy'%len(valid),valid)

def run_make_split1():

    df =  pd.read_csv('/root/share/project/kaggle/2019/steel/data/sample_submission.csv')
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    uid = df['ImageId'].unique().tolist()

    test = ['test_images/'+i for i in uid]
    np.save('/root/share/project/kaggle/2019/steel/data/split/test_%d.npy'%len(test),test)



def run_make_dummy():

    df = pd.read_csv('/root/share/project/kaggle/2019/steel/data/train.csv')
    df.fillna('', inplace=True)

    image_id =[
        '0007a71bf.jpg',
        '002fc4e19.jpg',
        '008ef3d74.jpg',
        '00ac8372f.jpg',
        '00bc01bfe.jpg', # *
        '00c88fed0.jpg',
        '00ec97699.jpg',
        '012f26693.jpg', # *
        '01cfacf80.jpg',
        '0391d44d6.jpg', # *
        'fff02e9c5.jpg', # *
        'ff6e35e0a.jpg',
        'ff73c8e76.jpg', # *
        'fec86da3c.jpg',
        'fea3da755.jpg',
        'fe2234ba6.jpg', # *
    ]

    image_id =[
        '012f26693.jpg', # *
        '0391d44d6.jpg', # *
        'fff02e9c5.jpg', # *
        'fe2234ba6.jpg', # *
    ]
    for i in image_id:
        print(i)

        rle = [
            df.loc[df['ImageId_ClassId']==i + '_1','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==i + '_2','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==i + '_3','EncodedPixels'].values[0],
            df.loc[df['ImageId_ClassId']==i + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread('/root/share/project/kaggle/2019/steel/data/train_images/%s'%(i), cv2.IMREAD_COLOR)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1) for r in rle])


        ##---
        step=300
        s = mask.sum(0).sum(0)
        v = [ -s[i: i+640].sum() for i in range(0,1600-640,step) ]
        argsort = np.argsort(v)

        #if 0:
        for k in range(2):
            t = argsort[k]

            print(-v[t])
            x0 = t*step
            x1 = x0+640

            dump_dir = '/root/share/project/kaggle/2019/steel/data/dump'
            os.makedirs(dump_dir+'/256x256/image',exist_ok=True)
            os.makedirs(dump_dir+'/256x256/mask',exist_ok=True)
            os.makedirs(dump_dir+'/256x512/image',exist_ok=True)
            os.makedirs(dump_dir+'/256x512/mask',exist_ok=True)
            os.makedirs(dump_dir+'/256x640/image',exist_ok=True)
            os.makedirs(dump_dir+'/256x640/mask',exist_ok=True)


            if 1:
                cv2.imwrite(dump_dir+'/256x640/image/%s_%d.png'%(i[:-4],k), image[:,x0:x1])
                np.save    (dump_dir+'/256x640/mask/%s_%d.npy'%(i[:-4],k), mask[...,x0:x1])
            if 1:
                cv2.imwrite(dump_dir+'/256x512/image/%s_%d0.png'%(i[:-4],k), image[:,x0:x0+512])
                np.save    (dump_dir+'/256x512/mask/%s_%d0.npy'%(i[:-4],k), mask[...,x0:x0+512])
                cv2.imwrite(dump_dir+'/256x512/image/%s_%d1.png'%(i[:-4],k), image[:,x1-512:x1])
                np.save    (dump_dir+'/256x512/mask/%s_%d1.npy'%(i[:-4],k), mask[...,x1-512:x1])
            if 1:
                cv2.imwrite(dump_dir+'/256x256/image/%s_%d0.png'%(i[:-4],k), image[:,x0:x0+256])
                np.save    (dump_dir+'/256x256/mask/%s_%d0.npy'%(i[:-4],k), mask[...,x0:x0+256])
                cv2.imwrite(dump_dir+'/256x256/image/%s_%d1.png'%(i[:-4],k), image[:,x1-256:x1])
                np.save    (dump_dir+'/256x256/mask/%s_%d1.npy'%(i[:-4],k), mask[...,x1-256:x1])


            #cv2.rectangle(image,(x0,0),(x1,256),(0,0,255),10)
        ##---

        overlay = np.vstack([m for m in mask])

        image_show('image',image,0.5)
        image_show_norm('mask',overlay,0,1,0.5)
        cv2.waitKey(1)



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_rle()

    #run_make_split1()
    run_make_dummy()


