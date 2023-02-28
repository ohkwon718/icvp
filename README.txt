0. Directory settings with ScenFlow dataset(https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
├── icvp (current)
    ├── code
├── dataset
    ├── SceneFlow
        ├── camera_data
        ├── disparity
        ├── frames_finalpass

1. To create conda environment with dependencies, please run
conda env create -f environment.yml
conda activate icvp

2. To train the model with Sceneflow dataset, please run
python run_train.py --path-run ./training/ft --path-target code

3. If you want to resume an early-stopped training, please run
python run_train.py --path-resume ./training/ft

4. To evaluate Sceneflow dataset with the trained model, please run
python run_eval.py --path-run ./training/ft

5. With the trained model, you can extract a disparity map from a pair of images by running
python run_predict.py --path-run ./training/ft --img-left <left-image-filename> --img-right <right-image-filename>
