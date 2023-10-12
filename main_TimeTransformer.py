from learner_func.learner_TimeTransformer import  trainer_TimeTransformer_base
from my_config import Config
if __name__ == '__main__':

    #
    # ranlist2 = Config.FOLD_ID_CHSZ
    # for myran in ranlist2:
    #     trainer_TimeTransformer_base(
    #         myran, patch_size=64, run_time=1, folds=5,
    #         model_name_base='WHC')
    #
    # ranlist2 = Config.FOLD_ID_CHSZ
    # for myran in ranlist2:
    #     trainer_TimeTransformer_base(
    #         myran, patch_size=64, run_time=1, folds=3,
    #         model_name_base='WHC')

    ranlist1 = Config.FOLD_ID_TUSZ
    for myran in ranlist1:
        trainer_TimeTransformer_base(
            myran, patch_size=64, run_time=1, folds=3,
            model_name_base='TUSZ')
