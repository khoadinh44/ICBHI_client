# ICBHI_client: 4 algorithm CNNs for ICBHI 2017 dataset
Preprocess:  Mel spectrogram

CNN type: EfficientNetV2M, MobileNetV2, InceptionResNetV2, ResNet152V2

## Important imformation in ArgumentParser of train.py:
data_dir: - For example (should be): /data/ICBHI_final_database
          - Description: Path to the original data.
          
save_data_dir: - For example (should be): /data/
               - Description: Path to the saved preprocessing data.
               
model_path: - For example (should be): /data/'model_'model_name'.h5
            - Description: Path to the saved weight
            
## Run train.py
          %cd /ICBHI_client
          !python train.py --data_dir /data/ICBHI_final_database --save_data_dir /data/ --model_path /data/ --model_name EfficientNetV2M 
               
