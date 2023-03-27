CURRENT="$PWD"
DUMPED_PATH="$CURRENT/dumped"
DATA_PATH="$CURRENT/data"
BACKBONE_FOLDER=${DUMPED_PATH}/backbones/continual/resnet18

/nfs4/anurag/ddpac/bin/python maml_trainer.py --trial pretrain \
                                --tb_path tb \
                                --data_root $DATA_PATH \
                                --classifier linear \
                                --model_path $BACKBONE_FOLDER/1 \
                                --continual \
                                --model resnet18 \
                                --no_dropblock \
                                --save_freq 100 \
                                --no_linear_bias \
                                --set_seed 5
