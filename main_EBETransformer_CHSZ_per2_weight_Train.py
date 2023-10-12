from learner_func.learner_EBE_Transformer_per2 import  trainer_data_augmentation_TimeTransformer_base
from my_config import Config
if __name__ == '__main__':
    ranlist = Config.generate_ranlist()

for myran in ranlist:
        trainer_data_augmentation_TimeTransformer_base(
            myran, patch_size=64, run_time=1, folds=3,
            l=4, mybeta=0.01, mylambda=1, weight_Type='Train',
            mmd= False, pearson=False,  temp= 6, is_L1=True,
            model_name_base='WHC')


