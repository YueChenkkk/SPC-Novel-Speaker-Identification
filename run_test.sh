MASTER_ADDR=localhost \
MASTER_PORT=11003 \
CUDA_VISIBLE_DEVICES=2 \
python test.py \
--world-size 1 \
--output-name test_on_wp \
--ckpt-dir /home/ychen/180/model/SpeakerIdentification/psi/train_wp_test_0 \
--data-dir ./data/wp_data \
--batch-size 4
