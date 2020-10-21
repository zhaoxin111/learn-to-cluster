import shutil
import os
from utils import misc
from misc import read_probs

def check_clustered_subjects(imgs_file,cluster_file,saved_folder,min_num=1):
    max_imgs = 5000
    misc.check_folder_exist(saved_folder)
    preds = [x[:-1] for x in open(cluster_file,'r').readlines()[:max_imgs]]
    imgs  = [x.split(' ')[0] for x in open(imgs_file,'r').readlines()[:max_imgs]]
    labels = set(preds)
    print('total {} subjects'.format(len(labels)))
    subjects_dict = {}
    for i, label in enumerate(preds):
        _subjects = subjects_dict.get(label,[])
        _subjects.append(imgs[i])
        subjects_dict[label]=_subjects
    subjects_dict = dict(sorted(subjects_dict.items(),key=lambda item: len(item[1]),reverse=True))
    count = 0
    for i, label in enumerate(subjects_dict.keys()):
        if len(subjects_dict[label])<min_num: continue
        count+=1
        for img in subjects_dict[label]:
            ori_subject_name = img.split('/')[-2]
            new_subject_name = '{}_{}'.format(label,ori_subject_name)
            path = os.path.join(saved_folder, new_subject_name)
            misc.check_folder_exist(path)
            shutil.copy(img,os.path.join(path,os.path.basename(img)))
    print('{} clusters has more than {} imgs'.format(count,min_num))

def get_part_renren_features():
    feature_path = "/data/zhaoxin_data/face_data/ms1m_data_featureBin_labelFile/ms1m-images_features_res100_256.bin"
    label_path = "/data/zhaoxin_data/face_data/ms1m_data_featureBin_labelFile/ms1m-images_list.txt"
    out_features_bin_file = "/data/zhaoxin_data/face_data/ms1m_data_featureBin_labelFile/ms1m_58W.bin"
    out_label_file = "/data/zhaoxin_data/face_data/ms1m_data_featureBin_labelFile/ms1m_58W.txt"
    out_num_imgs = 580000

    all_lines = open(label_path,'r').readlines()
    features = read_probs(feature_path,len(all_lines),256)
    part_features = features[:out_num_imgs,:]
    part_features.tofile(out_features_bin_file)

    with open(out_label_file,'w') as f:
        for line in all_lines[:out_num_imgs]:
            f.write(line.split(' ')[-1])

def sh1():
    root = '/data/zhaoxin_data/ms1m-images'
    subjects = glob(os.path.join(root,'*'))
    len_subjects = len(subjects)
    print(len_subjects)
    subject2index = {subject:index for subject, index in zip(subjects,range(len_subjects))}
    
    f = open('/data/zhaoxin_data/face_data/ms1m-images_list.txt','w')
    for subject in subjects:
        imgs = glob(os.path.join(subject,'*.jpg'))
        for img in imgs:
            content = '{} {}\n'.format(img,subject2index[subject])
            # print(content)
            f.write(content)

if __name__ == "__main__":
    check_clustered_subjects("/data/zhaoxin_data/face_data/test_cluster_data_labels.txt",\
        'data/work_dir/cfg_test_gcnv_ms1m/renren_50W_gcnv_k_80_th_0.0/latest_gcn_feat/tau_0.8_pred_labels.txt',\
        '/data/zhaoxin_data/GCN/sleftrained_renren_50W_res100_gvnv',1)

    # labels = open("/data/zhaoxin_data/face_data/test_cluster_data_labels.txt",'r').readlines()
    # with open('/home/zhaoxin/workspace/face/learn-to-cluster/data/labels/test_cluster_data_res50.meta','w') as f:
    #     for line in labels:
    #         f.write(line.split(' ')[1])

    # get_part_renren_features()