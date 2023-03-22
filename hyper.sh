# curl -o /etc/yum.repos.d/ceph_el7.repo http://gaia.repo.oa.com/ceph_el7.repo
# yum install ceph-fuse --enablerepo=ceph-nautilus -y
# mkdir /etc/ceph
# cp /apdcephfs/share_1363008/shared_info/joylv/ceph.client.aimmb.keyring /apdcephfs/share_1363008/shared_info/joylv/ceph.conf /etc/ceph;
# mkdir /mnt/group-ai-medical-cq;
# ceph-fuse /mnt/group-ai-medical-cq -r /aimmb -k /etc/ceph/ceph.client.aimmb.keyring -c /etc/ceph/ceph.conf -n client.aimmb;

# mkdir /mnt/group-ai-medical-cq
# mount -t ceph 9.130.225.224,9.130.225.252,9.130.225.199:/aimmb /mnt/group-ai-medical-cq -o name=aimmb,secret=AQDxYB9idaqnLRAAB1KZzxAnb3fhr7+q2PIuQw==

export http_proxy=http://9.86.9.201:3128
export https_proxy=http://9.86.9.201:3128
export no_proxy=mirrors.tencent.com
pip3 install timm==0.6.7 dataclasses==0.8

# l={0.0016, 0.0032, 0.0075, 0.015, 0.03, 0.045}
q=6
l=0.045
model=cheng2020-attn
data_type=vector
train_type=hyper
dataset_num=64
# cheng2020-anchor cheng2020-attn cheng2020-quant he2022-elic ours patent -FinalCut-UNet -q$q
MODEL_PATH=/mnt/group-ai-medical-cq/private/joylv/data/lora/${model}-$train_type-$data_type-compare/q$q-lambda${l}-$dataset_num/
# MODEL_PATH=/mnt/group-ai-medical-cq/private/joylv/data/train_frame_i_v2/
file=${MODEL_PATH}checkpoint.pth.tar

if [ -f "$file" ]; then
        echo 'checkpoint exist, load checkpoint'
        ckp="--checkpoint ${file}"
else
        echo 'mkdir'
        mkdir -p $MODEL_PATH 
fi

# /mnt/group-ai-medical-cq/private/joylv/data/BRACS/train/

CUDA_VISIBLE_DEVICES=5 python3 train_${train_type}.py \
        --lambda ${l} \
        --quality ${q} \
        -m ${model} \
        --epochs 10000 \
        -lr 1e-3 \
        --batch-size 64 \
        --cuda \
        --save \
        --model_prefix $MODEL_PATH \
        -d  /mnt/group-ai-medical-cq/private/joylv/data/bam_v2/$data_type/test \
        -td /mnt/group-ai-medical-cq/private/joylv/data/bam_v2/$data_type/test \
        -vd /mnt/group-ai-medical-cq/private/joylv/data/bam_v2/$data_type/test  \
        $ckp
