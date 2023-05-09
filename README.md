# 1) Download and install Python version 3.7
https://www.python.org/downloads/

# 2) Change into the <data_misc> folder and install all required packages with the following line:
```
pip install -r requirements.txt
```

# 3) Change into the <scripts> folder and execute following comands to train

### Federated Learning
  ```
nohup python train_FedAvg.py -model resnet18 -model_name resnet18 -subname FL_FedAvg -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 2 -epochs_unfrozen 8 -learning_rate 0.0017335209373335885 -sync_factor 0.2 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 300 -clinic clinic1 -participating_clinics clinic1_clinic2_clinic3_clinic4_clinic5 -leading_clinic clinic1 -testing_clinic clinic6 -random_erasing_probability 0.9390864270863725 -random_erasing_max_count 4 &

nohup python train_FedAvg.py -model resnet18 -model_name resnet18 -subname FL_FedAvg -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 2 -epochs_unfrozen 8 -learning_rate 0.0017335209373335885 -sync_factor 0.2 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 300 -clinic clinic2 -participating_clinics clinic1_clinic2_clinic3_clinic4_clinic5 -leading_clinic clinic1 -testing_clinic clinic6 -random_erasing_probability 0.9390864270863725 -random_erasing_max_count 4 &

nohup python train_FedAvg.py -model resnet18 -model_name resnet18 -subname FL_FedAvg -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 2 -epochs_unfrozen 8 -learning_rate 0.0017335209373335885 -sync_factor 0.2 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 300 -clinic clinic3 -participating_clinics clinic1_clinic2_clinic3_clinic4_clinic5 -leading_clinic clinic1 -testing_clinic clinic6 -random_erasing_probability 0.9390864270863725 -random_erasing_max_count 4 &

nohup python train_FedAvg.py -model resnet18 -model_name resnet18 -subname FL_FedAvg -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 2 -epochs_unfrozen 8 -learning_rate 0.0017335209373335885 -sync_factor 0.2 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 300 -clinic clinic4 -participating_clinics clinic1_clinic2_clinic3_clinic4_clinic5 -leading_clinic clinic1 -testing_clinic clinic6 -random_erasing_probability 0.9390864270863725 -random_erasing_max_count 4 &

nohup python train_FedAvg.py -model resnet18 -model_name resnet18 -subname FL_FedAvg -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 2 -epochs_unfrozen 8 -learning_rate 0.0017335209373335885 -sync_factor 0.2 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 300 -clinic clinic5 -participating_clinics clinic1_clinic2_clinic3_clinic4_clinic5 -leading_clinic clinic1 -testing_clinic clinic6 -random_erasing_probability 0.9390864270863725 -random_erasing_max_count 4 &
  ```

  
### Centralized
  ```
nohup python train_centralized.py -model resnet18 -model_name resnet18 -subname FL_centralized -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 9 -epochs_unfrozen 4 -learning_rate 0.0034512756844944955 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 200 -participating_clinics clinic1_clinic2_clinic3_clinic4_clinic5 -random_erasing_probability 0.03503048342527504 -random_erasing_max_count 6
  ```


### Centralized leave one out
  ```
nohup python train_centralized.py -model resnet18 -model_name resnet18 -subname FL_centralized_leave_one_clinic_out_CLINIC1_22_03_23 -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 9 -epochs_unfrozen 9 -learning_rate 0.007895373126386886 -minimal_training True -custom_random_state 0 -sample_rate_majority_class 400 -participating_clinics clinic2_clinic3_clinic4_clinic5 -random_erasing_probability 0.4268647044312376 -random_erasing_max_count 3  &

nohup python train_centralized.py -model resnet18 -model_name resnet18 -subname FL_centralized_leave_one_clinic_out_CLINIC2_22_03_23 -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 9 -epochs_unfrozen 9 -learning_rate 0.007895373126386886 -minimal_training True -custom_random_state 0 -sample_rate_majority_class 400 -participating_clinics clinic1_clinic3_clinic4_clinic5 -random_erasing_probability 0.4268647044312376 -random_erasing_max_count 3 &

nohup python train_centralized.py -model resnet18 -model_name resnet18 -subname FL_centralized_leave_one_clinic_out_CLINIC3_22_03_23 -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 9 -epochs_unfrozen 9 -learning_rate 0.007895373126386886 -minimal_training True -custom_random_state 0 -sample_rate_majority_class 400 -participating_clinics clinic1_clinic2_clinic4_clinic5 -random_erasing_probability 0.4268647044312376 -random_erasing_max_count 3 &

nohup python train_centralized.py -model resnet18 -model_name resnet18 -subname FL_centralized_leave_one_clinic_out_CLINIC4_22_03_23 -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 9 -epochs_unfrozen 9 -learning_rate 0.007895373126386886 -minimal_training True -custom_random_state 0 -sample_rate_majority_class 400 -participating_clinics clinic1_clinic2_clinic3_clinic5 -random_erasing_probability 0.4268647044312376 -random_erasing_max_count 3 &

nohup python train_centralized.py -model resnet18 -model_name resnet18 -subname FL_centralized_leave_one_clinic_out_CLINIC5_22_03_23 -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 9 -epochs_unfrozen 9 -learning_rate 0.007895373126386886 -minimal_training True -custom_random_state 0 -sample_rate_majority_class 400 -participating_clinics clinic1_clinic2_clinic3_clinic4 -random_erasing_probability 0.4268647044312376 -random_erasing_max_count 3 &
  ```

### Ensemble
  ```
nohup python train_centralized_ensemble.py -model resnet18 -model_name resnet18 -subname FL_centralized_ensemble -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 4 -epochs_unfrozen 2 -learning_rate 0.007962377303224796 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 100 -clinic clinic1 -random_erasing_probability 0.27335458998696716 -random_erasing_max_count 2  &

nohup python train_centralized_ensemble.py -model resnet18 -model_name resnet18 -subname FL_centralized_ensemble -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 4 -epochs_unfrozen 2 -learning_rate 0.007962377303224796 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 100 -clinic clinic2 -random_erasing_probability 0.27335458998696716 -random_erasing_max_count 2  &

nohup python train_centralized_ensemble.py -model resnet18 -model_name resnet18 -subname FL_centralized_ensemble -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 4 -epochs_unfrozen 2 -learning_rate 0.007962377303224796 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 100 -clinic clinic3 -random_erasing_probability 0.27335458998696716 -random_erasing_max_count 2  &

nohup python train_centralized_ensemble.py -model resnet18 -model_name resnet18 -subname FL_centralized_ensemble -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 4 -epochs_unfrozen 2 -learning_rate 0.007962377303224796 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 100 -clinic clinic4 -random_erasing_probability 0.27335458998696716 -random_erasing_max_count 2  &

nohup python train_centralized_ensemble.py -model resnet18 -model_name resnet18 -subname FL_centralized_ensemble -num_workers 8 -name_main_dataframe dummy_df.p -batch_size 32 -epochs_frozen 4 -epochs_unfrozen 2 -learning_rate 0.007962377303224796 -minimal_training False -custom_random_state 0 -sample_rate_majority_class 100 -clinic clinic5 -random_erasing_probability 0.27335458998696716 -random_erasing_max_count 2  &
  ```

