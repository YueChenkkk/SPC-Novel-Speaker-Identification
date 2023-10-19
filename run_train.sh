MASTER_ADDR=localhost \
MASTER_PORT=11002 \
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--world-size 1 \
--data-dir ./data/wp_data \
--epoch-num 50 \
--total-batch-size 8 \
--batch-size-per-gpu 4 \
--lr 1e-5 \
--lr-gamma 0.98 \
--early-stop 5 \
--margin 1.0 \
--max-len 512 \
--left-aux 1 \
--right-aux 1 \
--prompt-type 3 \
--role-mask-prob -1.0 \
--lbd1 0.3 \
--lbd2 -1.0 \
--pretrained-dir /home/ychen/180/pretrained/chinese-roberta-wwm-ext-large \
--ckpt-dir /home/ychen/180/model/SpeakerIdentification/psi/train_wp_test_0