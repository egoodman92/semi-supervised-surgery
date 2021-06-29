print("IMPORTING ACTIONS DATASET")

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle
import torch
import numpy as np
import os
import pandas as pd
import utils
import urllib.request
import cv2
from image import Resize, create_video_clip, augment_raw_frames, test_time_augment
from time import strftime
from time import gmtime
import pickle
import subprocess
import datetime


DEFAULT_CATEGORIES = ['cutting', 'tying', 'suturing', 'background']
SEGMENT_LENGTH = 5



def get_train_val_dfs(anns_path, quality_videos=True, ann_agree=True, k=0):
    df = pd.read_csv(anns_path)

    df = df.sort_values(by=['video_id', 'start_seconds'])

    # TODO: remove and update col names
    #df[df['labeler_2'].isnull()]['labeler_2'] = df[df['labeler_2'].isnull()]['labeler_1']
    df = df[~df['labeler_2'].isnull()]
    # Set label to labeler_3
    # df.loc[~df['labeler_3'].isnull(), 'labeler_2'] = df['labeler_3']
    # only select where labeler_3 matches labeler_2
    df = df[(df['labeler_3'].isnull()) | (df['labeler_2'] == df['labeler_3'])]
    df['label'] = df['labeler_2']
    df['category'] = df['label']
    df['video_name'] = df['video_id']

    videos = None
    #if quality_videos:
    #    videos = utils.videos_by_quality(['good', 'okay'])
    # local miniset
    #videos = ['0EeZIRDKYO4.mp4', '0FHE1jSN3w0.mp4', '0oxhRpSq850.mp4', '0sN_jPLnyIE.mp4', '0VMxXsqmxYg.mp4', '1AwmGrnk-Qs.mp4', '1dki4N24iP8.mp4', '1EBmECJP-Vk.mp4', '1gKMtSA6VCY.mp4', '1PwQrgU2Lic.mp4', '1yd_OrU2apY.mp4', '1YvFWU90eaE.mp4', '2ekkzoZ_uns.mp4', '2j-J2IiHLB8.mp4', '2XuCP7jeF7A.mp4', '357DW8IYYt8.mp4', '_35HaYVPBIA.mp4', '3Fx1EiO5iRY.mp4', '-3gFrKiC99I.mp4', '3Q5D4vdXOqc.mp4', '3Ql0fGVrQeA.mp4', '4uwsmL5TwGI.mp4', '5cscLtlF-P0.mp4', '5NWSWNFboIY.mp4', '_5OpGzoHEtM.mp4', '5T2o554MNo0.mp4', '61Le2IPCp6E.mp4', '6idNh90AdtA.mp4', '6ywlQxRzZGw.mp4', '7d19RiHEBjQ.mp4', '7Ku5P3nL8TA.mp4', '7RljEDWk4ZU.mp4', '7YXOJzIrKFM.mp4', '884CY5zcTr4.mp4', '8bhWGpkmWp8.mp4', '8CKX0j-OBl8.mp4', '8cT4CV1gkAA.mp4', '8eFljZ449SY.mp4', '8EhXEAp-uLI.mp4', '8LGGfKHtiPc.mp4', '8OK-_4Wx3QY.mp4', '8VPsnDDEN04.mp4', 'A42Nt7WsxTE.mp4', 'a4M6zT_PTQg.mp4', 'AE2FLFsmoeA.mp4', 'AGxuM1_H7KE.mp4', 'AjwzlmTvT8A.mp4', 'AK8N5NoWlAI.mp4', 'aqPrL_RJPC4.mp4', 'ASCSNOsM1xE.mp4', 'aSnUUpTgYW0.mp4', 'ATEuI_Y69CI.mp4', 'axVfF8m0LCU.mp4', 'AYt94DNtKXU.mp4', '_B0S8vqgKT8.mp4', 'bbI9l7j-LP4.mp4', 'BdLSt0VARKU.mp4', 'bfZWsulahkQ.mp4', 'bgWCyjuumn0.mp4', 'BHerDVFqrL8.mp4', 'bM0gMFbQJ-E.mp4', 'BssDb0s5cdA.mp4', 'BWGNKectNVA.mp4', 'BWrTbbZYozw.mp4', 'bxm902jT1Ok.mp4', 'c7zxUjQSusQ.mp4', 'CAd_n-qgRSI.mp4', 'Cczsz7JrUGU.mp4', 'cdYW87IozMA.mp4', 'CfFrwiwgniU.mp4', 'CkKu-f_HknQ.mp4', 'cpgMJ7KOVl8.mp4', 'D11S93t3QiU.mp4', 'd4-VN5A6_Ac.mp4', 'd8xavK1zIhY.mp4', 'daht-zR1G4s.mp4', 'dBRV4qa_qxk.mp4', '-DGQsmwXmLg.mp4', 'DHFI6YnauIM.mp4', 'dKF78zABOQs.mp4', 'dLap-hcBxc0.mp4', 'DM3aDFlbh9A.mp4', 'DpeAsOXVruw.mp4', 'dPvRrcSsc6Y.mp4', 'Dqk7k7pnlVM.mp4', 'dqVauLWgZ5k.mp4', 'e12tIDPDfwU.mp4', 'E8UPLNhr2B4.mp4', 'EgVI5o-fS6o.mp4', 'ejXLixqZVSs.mp4', 'eko_xMEYTKw.mp4', 'ekzZ9azx4N4.mp4', 'En4tbEsYKwI.mp4', 'Epicflt4WBQ.mp4', 'eQCNSx0P7LM.mp4', 'erUVkDi6KqY.mp4', 'EswP8VDC85s.mp4', 'EUdac6A9n60.mp4', 'EWra7VzEiS4.mp4', 'eYrOeMtk0ng.mp4', 'F5wtY4Ns4nA.mp4', 'feF1aRtVAvc.mp4', 'fHZ_0htxQ4k.mp4', 'Fk5WjTJdVUA.mp4', 'FMLfc2V18HY.mp4', 'FotC4hB7Y0c.mp4', 'frw2-77Iwf0.mp4', 'ftkqy5JBuSM.mp4', 'FUGhWj5iv70.mp4', 'G3fOG8CcKEw.mp4', 'g9lzOYtGLic.mp4', 'GG7IByPZKec.mp4', 'giVy41llq00.mp4', 'GJ5RwKonnms.mp4', 'GkKGPN4gado.mp4', 'gL_7N9tvgT4.mp4', 'GQQzVPySxPI.mp4', 'GT-FLlE95KU.mp4', 'GwHruH8trhg.mp4', 'hE7HSiip4ZA.mp4', 'hGZez6oxS9E.mp4', 'hKgkoj8kaHo.mp4', 'HKhfkcgE0ag.mp4', 'h-Ma9JnznLc.mp4', 'HP8pRJsWlGk.mp4', 'hqeq7pZOTgY.mp4', 'hQGTpK3CbpE.mp4', 'HSXoe4PkkWI.mp4', 'hSZOq0uqO6s.mp4', 'HW2LXjSoAc8.mp4', 'hyw7Ue6oW8w.mp4', 'i2GvmqhU4og.mp4', 'ifywgQA-0t8.mp4', 'iI09BTuD3xU.mp4', 'IJuourlriwc.mp4', 'ILZA2bFVGPE.mp4', 'iM5tFoqLX6Q.mp4', 'iONoQhdgIf0.mp4', 'IqpcBRQMOak.mp4', 'isyztOzq1Uw.mp4', 'iUwBjZUs_xo.mp4', 'IzVNghUkxI0.mp4', 'JjzptNEQJ2g.mp4', 'jkIGODwL228.mp4', 'jogJdtFU8SY.mp4', 'JT_ec9IVCzI.mp4', 'JUTyS7ZRRkQ.mp4', 'JuUFXltasaU.mp4', 'JvyO1LjGKhs.mp4', 'k0f8x5ZFnDs.mp4', 'K41gnfWPuhE.mp4', 'kcOqlifSukA.mp4', 'kHJh0MBx5Kc.mp4', 'KHrD5_zwTu8.mp4', 'kipTlXQpPZw.mp4', 'KlFRPypF1kg.mp4', 'kQTmL3C2dwE.mp4', 'kRWNgDQ-WcU.mp4', 'KUKObgS39yA.mp4', 'KV1bU0Kl9Fc.mp4', 'Kvpfsw4yCrM.mp4', 'KzPl-2e5Mw0.mp4', 'l5h_tOU_D9w.mp4', 'l7Fd7vTBkjE.mp4', 'L8k75Onag_o.mp4', 'lBmRC9LhFgw.mp4', 'LbqKFbnZvE4.mp4', '_lCgV4ZgAao.mp4', 'LgmXCOICHLA.mp4', 'LmkAisEsaew.mp4', 'lMmYj17LdbM.mp4', 'LPriMk5xy20.mp4', 'luDl5dLz1No.mp4', 'LVMCtRCJibI.mp4', 'M2PTmXVatbQ.mp4', 'mDvZwgwQ2oc.mp4', 'mMmYtpUschc.mp4', 'Mp2m1AMx8Ks.mp4', 'MQ0w6hXABDc.mp4', 'mq8pdoUFZ8g.mp4', 'mvYF35gw4Og.mp4', 'mYzL383plFw.mp4', 'N32N6VEcW2I.mp4', 'n5o16fYYp-8.mp4', 'N6Er2hAGB3Y.mp4', 'n6XWCpbmHtw.mp4', 'N8vXhfGUaHg.mp4', 'Nf5CArskkzM.mp4', 'njHyoJ_rKLs.mp4', 'NL72fNR2hSM.mp4', 'Nqs73GAXAxs.mp4', 'Nqyq0uaEBOg.mp4', 'NsXz7b--hC4.mp4', 'nwqyKvQ_mNk.mp4', 'nX4--aHdAkM.mp4', 'oD5gC2ESBnk.mp4', 'oDs_6O6AQjA.mp4', 'oewplXxSMrk.mp4', 'OMK_iNu3gIE.mp4', 'onVsHOnSypQ.mp4', 'opbgCSvNKfY.mp4', 'OrK0k3sBOHY.mp4', 'OsvR-ZxJtC0.mp4', 'OsYDOBpXC48.mp4', 'ou4iO5ah9ys.mp4', 'OXlUtv2DIzY.mp4', 'OZ9FXumhwFM.mp4', 'panFAJHLz_Q.mp4', 'pbkOtT3Q2AM.mp4', 'PC2SAKnlLok.mp4', 'PCyfP8I87AI.mp4', 'pHLuvSmOJRQ.mp4', 'pkvyOtGIbV8.mp4', 'pN5bKT7U_OQ.mp4', 'PqGpbc330j8.mp4', 'P-qQJChzlhM.mp4', 'PsYnEXGxf-M.mp4', 'pWIti7kfTyk.mp4', 'pwWamkMfIR0.mp4', 'PyIDO9qA2_g.mp4', 'PzaCNn8IIVw.mp4', 'pZNG2IN5ixo.mp4', 'qaAoYqg2vWo.mp4', 'QbOhP0qTXM4.mp4', 'QjYdLDYs4JI.mp4', 'qxaOo9Q0g2M.mp4', 'qz_dOdnqAnk.mp4', 'RCU-R4p16SU.mp4', 'RDpE1Q9Bkno.mp4', 'rmcomZXiud4.mp4', 'rtJzPBn6FWw.mp4', 'Rw5idx6VjbI.mp4', 'RWHBTwfa5C8.mp4', 'rzKGRcXNXm4.mp4', 's0z0WYcC7dA.mp4', 'S1R95eOuSNk.mp4', 'S4p9MmTfVTs.mp4', 'sayim_KU_8c.mp4', 'sFZchKZ10bc.mp4', 'sHZWvYrerEs.mp4', 'sIr0IxtC7C4.mp4', 'sL5tVxz89i4.mp4', 'SNsUtI82de8.mp4', 'SRQb7RGQ52M.mp4', 'StuUjtXj6u8.mp4', 'SublDZXJ7p4.mp4', 'sWfWnFk1W70.mp4', 'sWxGr_rj7rM.mp4', 'synW6molzgA.mp4', 'Sz57XML7IGA.mp4', 'sZzcnDjeUKg.mp4', 'T03GIXho9mY.mp4', 'T5sNqOFwTfQ.mp4', 'tBfZ8UkhAG4.mp4', 'tbxSEoWwqCM.mp4', 'TcYgRmsw_jg.mp4', 'tE8Ml8aDk18.mp4', 'TFwFMav_cpE.mp4', 'Tg3Pg6f-mjg.mp4', 'thgXniYmwII.mp4', 'tnXv-7fulTY.mp4', 'toE_4MtsqQM.mp4', 'tOLu-p5V-7o.mp4', 'txCYSkZIrjE.mp4', 'TZ9IgsIMRW4.mp4', 'UDaMvSGMa3w.mp4', 'uGT08nyk7qA.mp4', 'UJzc5o-rXOM.mp4', 'unQH_a-Rm-I.mp4', 'urMq4AtExDg.mp4', 'USL06hfnyKg.mp4', 'uVvBlWpwopw.mp4', 'U_vxJyjmEZ0.mp4', 'Uy2fFSuYfpE.mp4', 'UyB951vsUfA.mp4', 'UyTv72j6Lno.mp4', 'V3Fd-cRcdCA.mp4', 'V6pL0fMsVn0.mp4', 'V7vkRKaUkn8.mp4', 'vfHOapUt5kA.mp4', 'VFyJ65hEF3k.mp4', 'V_kW8dMjGX0.mp4', 'VL_UNwlWd3c.mp4', 'vmJjIIt-wAY.mp4', 'Vnab8vZQcK8.mp4', 'VsKw5d-4rq8.mp4', 'VtJtGtC3R80.mp4', 'vtK1XN4ZaU4.mp4', 'VvkCaxXqFfY.mp4', 'VvTrKCFtZhA.mp4', 'VyN4c_wsZuY.mp4', 'W8ZBbjZj74M.mp4', 'wBADkB_PFkA.mp4', 'WDLZ-G0IRJQ.mp4', 'wDu9VyqfNu8.mp4', 'wEivan1FAIA.mp4', 'wIr-9Myre4k.mp4', 'wiT6R0xxh7w.mp4', 'wIVcst7zrig.mp4', 'WJ2jS88EUmo.mp4', 'wmDkmO4PAYI.mp4', 'WR8G3F2TDxI.mp4', 'WTO1F3NG8PE.mp4', 'wVszq_SYrAQ.mp4', 'wy7Q_xUlwHI.mp4', 'wZ2LvPEP84g.mp4', 'wZTMcbt85J4.mp4', 'x301dRwjBt0.mp4', 'x98rpVdG96Y.mp4', 'XfnUomzjvfs.mp4', 'Xg6vD3vngLQ.mp4', 'xh9XtKhfzKU.mp4', 'XnkQgjOM5RU.mp4', 'XoGw0yrEv4E.mp4', 'XZleSz21hik.mp4', 'y0ie8xT6Hx0.mp4', 'ycYdlwoKtEw.mp4', 'YFs_AfPeK84.mp4', 'YGJLGKetzy0.mp4', 'Yl3S1UtzmFI.mp4', 'YLz8sAnLDM8.mp4', 'ytgWAMS1SkE.mp4', 'YXAGt6GtFwE.mp4', 'Z4Z03ebdSa8.mp4', 'ZAwOzpqlZkA.mp4', 'zJkQP-BFl8c.mp4', 'zmQpJvYVsi4.mp4', 'zNnwVj9whYA.mp4', 'zOHKqmSAZto.mp4', 'ZP58Wx0QaWE.mp4', 'zPP8sy1C6-4.mp4', 'zRk7vi3FsyU.mp4', 'ZwdbtYzY_6s.mp4', 'ZXQPWRUdFqc.mp4', 'zyW8m35aEr0.mp4', 'ZyWXlbQ8XdI.mp4']
    #videos = ['0EeZIRDKYO4.mp4', '0FHE1jSN3w0.mp4', '0oxhRpSq850.mp4']
    # if ann_agree:
    #     df = utils.segments_by_nonexpert_agreement(df)
    #print("LENGTH OF VIDEOS IS", len(videos))
    t_df, v_df = get_video_split(df, videos, k)
    #print("Got dataset", t_df.shape, v_df.shape)
    #print(t_df.head)
    return t_df, v_df


def balance_classes(df, categories):
    print("We are balancing categories", categories)
    print("\nAbout to balance classes!", df.shape)
    min_count = 1e10
    for category in categories:
        count = len(df.loc[df['category'] == category])
        if count < min_count:
            min_count = count
    max_count = min_count

    print("Found max common action count was", max_count)

    df2 = pd.DataFrame()
    for category in categories:
        df2 = df2.append(df.loc[df['category'] == category][:max_count])

    print("Balanced dataset!", df2.shape, "\n")
    return df2

def get_category_based_split(df):
    categories = SurgeryDataset.categories
    df = balance_classes(df, categories)
    num_categories = len(categories)
    split = 0.8
    train_count = int(split * len(df))
    train_per_category_count = int(train_count / num_categories)
    t_df = pd.DataFrame()
    for c in categories:
        tmp_df = df[df['category'] == c].head(train_per_category_count)
        t_df = t_df.append(tmp_df)
    v_df = df[~df['filename'].isin(t_df['filename'])]
    return t_df, v_df


def get_video_split(df, videos=None, k=0):
    if videos is None:
        videos = sorted(list(set(pd.read_csv('data/train.csv')['video_id'])))
    print("Study includes {} videos found in data/train.csv".format(len(videos)))
    folds = 7
    fold_size = int(len(videos) / folds)
    fold_index = fold_size * k
    #print("fold index: %d" % fold_index)
    videos = videos[fold_index:] + videos[:fold_index]
    
    #print(videos[0:5])

    train_val_split = 1.0 - (1.0 / folds)
    train_count = int(train_val_split * len(videos))
    train_videos = videos[0:train_count]
    val_videos = videos[train_count:]
    #print("Train videos: %d, val videos: %d" % (len(train_videos), len(val_videos)))

    t_df = df[df['video_name'].isin(train_videos)]
    v_df = df[df['video_name'].isin(val_videos)]
    if len(t_df) == 0 or len(v_df) == 0:
        train_segs = int(train_val_split * len(df))
        t_df = df.iloc[0:train_segs]
        v_df = df.iloc[train_segs:]

    # t_df = t_df.sample(2000)
    # v_df = v_df.sample(400)

    return t_df, v_df



class SurgeryDataset(Dataset):
    categories = None

    def __init__(self, df, data_dir='data/', mode='train', model='BLV', balance=True,
                 pre_crop_size=256, aug_method='04-20', segment_length=5):

        self.data_dir = data_dir
        self.categories = SurgeryDataset.categories
        if balance:
            self.df = balance_classes(df, self.categories)
        else:
            self.df = df[df['category'].isin(self.categories)]
        self.mode = mode
        self.model = model

        self.pre_crop_size = pre_crop_size
        self.segment_length = segment_length
        self.aug_method = aug_method
        # TODO: Add transformations

    @staticmethod
    def raw_frames_to_input(raw_frames, num_segments=8, method='multi-scale'):
        if method == 'BLV':
            input = raw_frames
            input = np.asarray(raw_frames, dtype=np.uint8)
        elif method == 'multi-scale':
            input = np.asarray(raw_frames[0::8], dtype=np.uint8)
            input = np.expand_dims(input, axis=0)
            for i in range(1, 2):
                new_input = np.asarray(raw_frames[i*2::8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
            for i in range(2):
                new_input = np.asarray(raw_frames[0::4][i*8:i*8 + 8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
            for i in range(4):
                new_input = np.asarray(raw_frames[0::2][i*8:i*8 + 8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
        else:  # uniform
            input = np.asarray(raw_frames[0::8], dtype=np.uint8)
            input = np.expand_dims(input, axis=0)
            for i in range(1, num_segments):
                new_input = np.asarray(raw_frames[i::8], dtype=np.uint8)
                new_input = np.expand_dims(new_input, axis=0)
                input = np.concatenate((input, new_input), axis=0)
        return input

    def remote_load_frame(self, video_id, filename):
        img_size = 224
        remote_url = ("https://marvl-surgery.s3.amazonaws.com/frames/%s/%s" % (video_id, filename))
        #print("Remote URL: %s" % remote_url)
        with urllib.request.urlopen(remote_url) as req:
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # 'Load it as it is'
            if not (img.shape[0] == img_size and img.shape[1] == img_size):
                img = Resize(dsize=(img_size, img_size)).process(img)
            return img

    def get_frames(self, video_id, start_frame, end_frame, num_frames=64):
        # TODO: set as env variable
        #frames_dir = os.path.join(self.data_dir, "frames")
        frames_dir = "/pasteur/data/YoutubeSurgery/images-fps_15"
        if end_frame - start_frame < num_frames:
            print("Warning: not enough frames in segment")
        frames = []
        for i in range(start_frame, start_frame + num_frames):
            filename = (video_id + "-" + "%.9d.jpg") % i
            file_path = os.path.join(frames_dir, video_id, filename)
            if os.path.exists(file_path):
                img = cv2.imread(file_path)[:, :, [2, 1, 0]]
            else:
                img = self.remote_load_frame(video_id, filename)
            frames.append(img)
        return frames

    '''
    def get_frames_from_video(self, video_id, video_path, start_seconds, end_seconds, num_frames=16, cache=False):
        num_frames=16
        file_path = video_path

        count = 0

        frames = []

        cap = cv2.VideoCapture(file_path)

        playback_fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_seconds*playback_fps)

        meann = np.array([[[0.4115, 0.3570, 0.3051]]]) #emmetts calculated mean on surgery dataset
        stdd = np.array([[[0.3311, 0.2877, 0.2650]]]) #emmetts calculated mean on surgery dataset

        #print("start_seconds, end_seconds, fps, start_frame", start_seconds, end_seconds, playback_fps, start_frame)
        for i in range(num_frames):
            frame = int(start_frame + i)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, img = cap.read()
            if not (img is None or success == False):
                img = 255*(((img.astype(np.float32)/255) - meann) / stdd) #np.float32
                frames.append(img)
                #frames.append(img.astype(np.uint8))
                #frames.append(img)
                count += 1
            else:
                # TODO: use zero padded image, or fix annotations
                print("WARNING: frame: %d for  filepath %s, start seconds %d, is None" % (i, file_path, start_seconds))
            if count >= num_frames:
                break
        if cache:
            print("Caching to", self.cached_path(video_id, start_seconds))
            pickle.dump(frames, open(self.cached_path(video_id, start_seconds), "wb"))
        return frames

    '''
    #WILLS VERSION BEFORE EMMETT MODIFIED IT (ALTHOUGH I DID STILL CHANGE 64 -> 16)!
    def get_frames_from_video(self, video_id, video_path, start_seconds, end_seconds, num_frames=16, cache=True):
        #frame_rate = 15
        frame_rate = int(float(num_frames) / (end_seconds - start_seconds))
        file_path = video_path

        count = 0
        frames = []
        start_frame = frame_rate * start_seconds
        cap = cv2.VideoCapture(file_path)
        playback_fps = cap.get(cv2.CAP_PROP_FPS)
        for i in range(num_frames):
            ratio = (float(playback_fps)/frame_rate)
            frame = int((start_frame + i) * ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, img = cap.read()
            if not (img is None or success == False):
                img = Resize(dsize=(self.pre_crop_size, self.pre_crop_size), rel_scale='max').process(img)
                frames.append(img)
                count += 1
            else:
                # TODO: use zero padded image, or fix annotations
                print("WARNING: frame: %d for  filepath %s, start seconds %d, is None" % (i, file_path, start_seconds))
            if count >= num_frames:
                break
        if cache:
            print("Caching frames to", self.cached_path(video_id, start_seconds))
            pickle.dump(frames, open(self.cached_path(video_id, start_seconds), "wb"))
        return frames






    def cached_dir(self):
        return os.path.join(self.data_dir, "cached-%d-%d" % (self.pre_crop_size, self.segment_length))

    def cached_path(self, video_id, start_seconds):
        return os.path.join(self.cached_dir(), "%s-%f.pkl" % (video_id, start_seconds))

    def cached_frames(self, record):
        data_path = self.cached_path(record['video_id'], record['start_seconds'])
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                return data
        return None

    def video_path(self, video_id):
        return self.data_dir + video_id + ".mp4"

    def download_clip(self, video_id, start_seconds, end_seconds):
        print("IN DOWNLOAD CLIP!")
        remote_video_path = "https://www.youtube.com/watch?v=" + video_id
        local_path = self.data_dir + video_id + '-' + str(start_seconds) + '-' + str(end_seconds) + '.mp4'
        start = datetime.timedelta(seconds=start_seconds)
        duration = datetime.timedelta(seconds=(end_seconds - start_seconds))
        command = ("ffmpeg -y -ss %s -i $(youtube-dl -f 22 -g '%s') -t %s -c copy %s > /dev/null 2>&1" % (str(start), remote_video_path, str(duration), local_path))
        process = os.system(command)
        success = process == 0
        return local_path, success

    def zero_frames(self, num_frames=64):
        img_array = []
        for i in range(num_frames):
            img = np.zeros((self.pre_crop_size, self.pre_crop_size, 3))
            img_array.append(img)
        return img_array

    def frames_labels_meta(self, index):
        if not os.path.exists(self.cached_dir()):
            os.mkdir(self.cached_dir())
        record = self.df.iloc[index]
        labels = None
        raw_frames = self.cached_frames(record)
        video_id = record['video_id']
        #start_seconds = int(record['start_seconds'])
        #end_seconds = int(record['end_seconds'])
        start_seconds = record['start_seconds']
        end_seconds = record['end_seconds']

        if not raw_frames:
            #print("Video ID is", video_id)
            video_path = self.video_path(video_id)
            if os.path.exists(video_path):
                raw_frames = self.get_frames_from_video(video_id, video_path,
                                                    start_seconds, end_seconds)
            else:
                print("ERROR, NEED TO DOWNLOAD")
                local_path, success = self.download_clip(video_id, start_seconds, end_seconds)
                if success:
                    raw_frames = self.get_frames_from_video(video_id, local_path,
                                                    0, start_seconds - end_seconds)
                    print(local_path)
                else:
                    print("WARNING: could not download")
                    raise Exception('Could not download video %s' % video_id)
                    # raw_frames = self.zero_frames()
                    # print("WARNING: created zero frames")


        metadata = record
        #print("DURP", len(raw_frames), raw_frames[0].shape, labels)
        return (raw_frames, labels), metadata

    def __getitem__(self, index):
        (raw_frames, labels), record = self.frames_labels_meta(index)
        if self.mode == 'train':
            raw_frames, scale = augment_raw_frames(raw_frames, method=self.aug_method)
        else:
            raw_frames, scale = augment_raw_frames(raw_frames)




        #input = SurgeryDataset.raw_frames_to_input(raw_frames, method=self.model)

        # Normalize data between range [-1, 1]

        input = (raw_frames/255).astype(np.float32)


        #if self.model == 'BLV':
        #    if len(input.shape) == 5:
        #        input = input.transpose([0, 1, 4, 2, 3])
        #        input = np.reshape(input, (-1, 16*3, 224, 224))
        #    else:
        #        input = input.transpose([0, 3, 1, 2])
        #        input = np.reshape(input, (-1, 16*3, 224, 224))

        #this is the one magical piece needed to flip RBG channels so colors come out right!
        input = input[:, :, :, ::-1] - np.zeros_like(raw_frames)

        input = torch.from_numpy(input)
        input = input.permute(0, 3, 1, 2)
        #else:  # Convert to TSN input format
        #    input = input.transpose([1, 0, 4, 2, 3])
        #    input = torch.from_numpy(input)

        # Convert to one label per segment.
        #labels = (np.average(labels, axis=1) > 0.5).astype(np.float32)
        #labels = torch.from_numpy(labels)

        labels = np.zeros(len(self.categories), dtype=np.float32)
        labels[self.categories.index(record['category'])] = 1
        labels = torch.from_numpy(labels)

        if torch.cuda.is_available():
            input = input.cuda()
            labels = labels.cuda()

        record_id = index

        #have to send out input with these brackets because opencv images have different
        #RGB chanels than skimage images...
        return (input, record_id, labels, scale)

    def __len__(self):
        return len(self.df)
