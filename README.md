# MS-ATCNN
This is the code work for: ''Mobility-Supporting ATCNN in HLWNets''

# This final code version will be organized upon the acceptance

**Dataset collection file**: Optimal_UTI_collection.py; RNN_UTI_collection.py (for temporal dataset case);

**Training file**: ATCNN_train_loss.py, UTI_training_test.py;

**Model setup file**: ATCNN_model.py

**Important sub-functions**: mytopo.py, utils.py

The trained MSNN and RNN models are saved in the folder: /tracking_data/2024-......-MSNN-TypeX/final_model.pth and /tracking_data/2024-......-RNN/final_model.pth;
The parameters for models are saved in the folder: /tracking_data/2024-......-MSNN-TypeX/Hyper-parameters.json;

The structure of RNN can be detailed using the RNN model initialization in ATCNN_model.py, and Hyper-parameters values in /.../Hyper-parameters.json files. 
In general, the sliding window size of RNN is 10 (1 for 10 ms), input size is 3, output size is 1, layer number is 3, and dropout ratio is 0.5. Other parameters are listed in the /.../Hyper-parameters.json file.



