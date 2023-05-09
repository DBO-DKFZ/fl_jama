import comet_ml
import pickle, os, glob, sys, torch, random, gc, fastai, optuna, logging, configparser
from fastai.vision.all import *
from fastai.vision.augment import _slice
import sklearn.metrics as skm
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from optuna.integration import FastAIPruningCallback
from misc_funcs import *
from optuna.samplers import TPESampler
              
    
# args
parser = ArgumentParser()
parser.add_argument('-model', default='resnet18', type=str)
parser.add_argument('-model_name', default='resnet18', type=str)
parser.add_argument('-subname', default='test', type=str)
parser.add_argument('-num_workers', default=6, type=int)
parser.add_argument('-name_main_dataframe', default="df_main.p", type=str)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-epochs_frozen', default=1, type=int)
parser.add_argument('-epochs_unfrozen', default=1, type=int)
parser.add_argument('-learning_rate', default=2e-03, type=float)
parser.add_argument('-sync_factor', default=0.5, type=float)
parser.add_argument('-minimal_training', default='False', type=str)
parser.add_argument('-custom_random_state', default=32, type=int)
parser.add_argument('-sample_rate_majority_class', default=50, type=int)
parser.add_argument('-clinic', default='Berlin', type=str)
parser.add_argument('-participating_clinics', default='Berlin_Muenchen', type=str)
parser.add_argument('-leading_clinic', default='Berlin', type=str)
parser.add_argument('-testing_clinic', default='Dresden', type=str)
parser.add_argument('-random_erasing_probability', default=2e-03, type=float)
parser.add_argument('-random_erasing_max_count', default=2e-03, type=float)


args = parser.parse_args()


### Local paths
path_data_misc = Path("../data_misc/")
path_data = Path("../data/")
path_models = Path("../models/")
path_results = Path("../results/")
path_swarm = Path("../swarmCallback/")


### define some parameters
model_dict = dict({"resnet18":resnet18, "resnet34":resnet34, "resnet50":resnet50})
model = model_dict[args.model]
model_name = args.model_name
subname = args.subname
prefix = model_name + "_" + subname + "_" + args.clinic + "_"
custom_random_state = (None if args.custom_random_state==0 else args.custom_random_state)
clinic = args.clinic
leading_clinic = args.leading_clinic
testing_clinic = args.testing_clinic
is_sync_node = (True if clinic==leading_clinic else False)
clinic_list = args.participating_clinics.split("_")

df_main_path = f"{path_data_misc}/{args.name_main_dataframe}"
minimal_training = (True if args.minimal_training=="True" else False)
batch_size = args.batch_size
num_workers = args.num_workers
image_size = 224
sample_rate_majority_class = args.sample_rate_majority_class
epochs_frozen = args.epochs_frozen
epochs_unfrozen = args.epochs_unfrozen
learning_rate = args.learning_rate
sync_factor = args.sync_factor
path_to_config = path_swarm/f"tmp_swarm_communication/{prefix.replace(clinic, '')}config.conf"
config = configparser.ConfigParser()

### log file handling
old_stdout = sys.stdout
log_file = open(path_results/(prefix+"log_file.txt"), "w")
sys.stdout = log_file

# print(f"Let's Get This Party Started! with {clinic}")

def training_procedure(sample_rate_majority_class, epochs_frozen, epochs_unfrozen, learning_rate, sync_factor, random_erasing_max_count, random_erasing_probability, experiment):
       
    ### create dataframes
    pd.options.mode.chained_assignment = None  # default='warn'
    df_main = pickle.load(open(df_main_path, "rb"))
    ### retrieve all participating clinics from df
    df_train = df_main[df_main["clinic"].apply(lambda x:x in clinic_list)]
    ### create validation set from all participating clinics
    df_train = get_random_validation_set(df_train, percentage_valid=30, lbls=["0", "1"], clinics=clinic_list)
    ### remove all entries non-validation entries that are not the main clinic
    df_train = df_train[(df_train["clinic"] == clinic) | (df_train["is_valid"] == True)]
    
    df_train = df_train.sort_values(by=["is_valid", "clinic", "slide_name", "tile_path"], ascending=True).reset_index(drop=True)
    tile_count_list_train = df_train[df_train["is_valid"] == True].groupby(["slide_name"], sort=False).count().tile_path.values
    tile_slide_list_train = df_train[df_train["is_valid"] == True].slide_name.unique()
    
    # print("This should print my dataframe!")
    # print(df_train[df_train["is_valid"] == True].slide_name.unique())
    # print(df_train.drop_duplicates("slide_name").groupby(["is_valid", "label"]).count())
    

    slides_per_clinic_dict = df_main.drop_duplicates("slide_name").groupby("clinic").count().slide_name.to_dict()
    slides_per_state_dict = dict()
    for i in clinic_list:
        slides_per_state_dict[f"{path_swarm}/tmp_state_dicts/{model_name}_{i}_{subname}_state_dict.pkl"] = slides_per_clinic_dict[i]
        # print(f"{path_swarm}/tmp_state_dicts/{model_name}_{i}_{subname}_state_dict.pkl has{slides_per_clinic_dict[i]} slides!")

    if minimal_training:
        df_train = limit_number_of_tiles(df_train, "is_valid", False, 10)  
        df_train = limit_number_of_tiles(df_train, "is_valid", True, 10)
        tile_count_list_train = df_train[df_train["is_valid"] == True].groupby(["slide_name"], sort=False).count().tile_path.values
        sample_rate_majority_class=5

    tfms = [MyRandomErasing(p=random_erasing_probability, max_count=random_erasing_max_count)]


    def get_kwargs(**kwargs): return kwargs

    ### create DataLoaders
    dataloader_args = get_kwargs(df=df_train, image_size=image_size, path_data=path_data, batch_size=batch_size, balancer=True,
                                 sample_rate_majority_class=sample_rate_majority_class, tfms=tfms, location_col_name="tile_path",
                                 label_col_name="label", splitter_col_name="is_valid", num_workers=num_workers,
                                 slide_id_col_name="slide_name", use_custom_random_state=custom_random_state, shuffle=True)
    
    # print(f"These are my dataloader_args: {dataloader_args['df'].groupby(['is_valid', 'label']).count()}")

    dls, _ = create_dataloader(**dataloader_args)

    
    ### define metrics
    my_metrics = []
    my_metrics.append(SlideLevelSensitivity(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train))
    my_metrics.append(SlideLevelSpecificity(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train))
    my_metrics.append(SlideLevelBalAcc(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train))
    my_metrics.append(SlideLevelAuroc(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train, get_pred_per_slide=True))
    
    ### manage indexing for different sets
    tmp = df_train[df_train["is_valid"]==True].reset_index(drop=True)
    
    for current_clinic in df_train.clinic.unique():
        tmp_first_idx = min(tmp[(tmp["clinic"] == current_clinic) & (tmp["is_valid"] == True)].index.values)
        tmp_last_idx = max(tmp[(tmp["clinic"] == current_clinic) & (tmp["is_valid"] == True)].index.values)+1
        # my_metrics.append(SlideLevelSensitivity(tile_count_list=tile_count_list_train, from_to_idx=[tmp_first_idx,tmp_last_idx], name=f"slide_lvl_sens_{current_clinic}"))
        # my_metrics.append(SlideLevelSpecificity(tile_count_list=tile_count_list_train, from_to_idx=[tmp_first_idx,tmp_last_idx], name=f"slide_lvl_spec_{current_clinic}"))
        # my_metrics.append(SlideLevelBalAcc(tile_count_list=tile_count_list_train, from_to_idx=[tmp_first_idx,tmp_last_idx], name=f"slide_lvl_balacc_{current_clinic}"))
        my_metrics.append(SlideLevelAuroc(tile_count_list=tile_count_list_train, tile_slide_list=tile_slide_list_train, from_to_idx=[tmp_first_idx,tmp_last_idx], name=f"slide_lvl_auroc_{current_clinic}"))



    ### define callbacks
    my_cbs = [SamplerCallback(**dataloader_args),
              
              SwarmCallback(sync_factor=sync_factor, state_dicts_dir=f"{path_swarm}/tmp_state_dicts/", communication_dir=f"{path_swarm}/tmp_swarm_communication/",
                            is_sync_node=is_sync_node, model_name=model_name, subname=subname, clinic=clinic, clinic_list=clinic_list, slides_per_state_dict=slides_per_state_dict)]
    

    ### create learner
    learn = vision_learner(dls, model, metrics=my_metrics, cbs=my_cbs, model_dir=path_models).to_fp16()
    
    setattr(learn, "pred_per_slide", dict())

    ### train the model
    with learn.no_bar():
        learn.fine_tune(freeze_epochs=epochs_frozen, epochs=epochs_unfrozen, base_lr=learning_rate)
    
    learn._set_progress_to_finished()
    
    if is_sync_node:
        torch.save(learn.model.state_dict(), path_models/(prefix.replace(leading_clinic, "")+"final_state_dict.pkl"))
        learn.export(path_models/(prefix.replace(leading_clinic, "")+"export.pkl"))
    return None
    
def check_for_loaded_config():
    config_signal_list = []
    for c in clinic_list:
        if c != leading_clinic:
            config_signal_list.append(path_swarm/f"tmp_swarm_communication/{prefix.replace(clinic, c)}loaded")
    
    all_loaded = True
    for c in config_signal_list:
        if os.path.isfile(c) == False:
            all_loaded = False
    
    return all_loaded, config_signal_list


training_procedure(args.sample_rate_majority_class, args.epochs_frozen, args.epochs_unfrozen, args.learning_rate, args.sync_factor, args.random_erasing_max_count, args.random_erasing_probability, None)

sys.stdout = old_stdout
log_file.close()


if is_sync_node:
    do_del_paths = glob.glob(f"{str(path_swarm)}/tmp_swarm_communication/{model_name}*_{subname}*")
    for i in do_del_paths:
        os.remove(i)
