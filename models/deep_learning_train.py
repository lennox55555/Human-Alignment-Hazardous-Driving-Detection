import cv2
import pandas as pd
import numpy as np
import pickle
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from PIL import Image
from tqdm import tqdm
import os

class VideoDataset(Dataset):
    '''
    A Dataset class to pass both video frames and class labels as input to a model
    '''
    def __init__(self, df, labels, frames_dict):
        self.df = df # the original dataframe that links indexes to to keys in frames_dict
        self.labels = labels # the class labels
        self.frames_dict = frames_dict # dictionary linking dataframe rows to to video frames

    def __len__(self):
        return self.df.shape[0] #number of training samples

    def __getitem__(self, idx):
        clip_frame_id = self.df.iloc[idx]['clip_frames_id'] #frame index
        label = self.labels.iloc[idx] #label

        return self.frames_dict[clip_frame_id], label # clip frames and label

    def get_labels(self):
        return self.labels.values #all labels for this dataset
    

MODEL_CHECKPOINT = "deepLearningModel.pkl"

class DeepLearningModel():
    def __init__(self, data_dir, model_checkpoint_dir, model_checkpoint="deepLearningModel.pth", binned_video_file='processed/binned_video_dat_wo_user.csv', driving_video_dir='processed/driving_videos'):
        self.data_dir = data_dir
        self.binned_video_csv = os.path.join(data_dir, binned_video_file)
        self.videos_dir = os.path.join(data_dir, driving_video_dir)
        self.output_dir = os.path.join(self.data_dir, 'output')
        self.df = pd.read_csv(self.binned_video_csv, index_col=0)
        self.frames_dict = {}
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_checkpoint = os.path.join(model_checkpoint_dir, model_checkpoint)

        if not os.path.exists(self.model_checkpoint):
            self.df, self.frames_dict = self.add_frames_to_df(self.df)
        self.model = None
        self.test_loader = None

        
        os.makedirs(self.output_dir, exist_ok=True)
        

    def get_class_weights(self, loader):
        '''
        Get class weights from the loader

        Inputs:
            - loader (DataLoader): the loader containing labels
            - device: the gpu/cpu being used
        
        Returns:
            - (torch.FloatTensor): The class weights to be used
        '''
        labels = loader.dataset.get_labels()
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights).to(self.device)

    def get_class_weights_from_labels(self, labels):
        '''
        Get class weights from the list of labels

        Inputs:
            - labels (List): the list containing labels
            - device: the gpu/cpu being used
        
        Returns:
            - (torch.FloatTensor): The class weights to be used
        '''
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights).to(self.device)
        
    def training_loop(self, model, loader, epochs=10):
        '''
        The training loop for the model

        Inputs: 
            - model: the 3D computer vision model used for training
            - loader: the train loader
            - device: the gpu/cpu being used for trainng
            - epochs: the number of epochs used for training
        '''
        # Class Weights
        class_weights = self.get_class_weights(loader)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Move model to device and set to training mode
        model.to(self.device)
        model.train() 

        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0

            for clips, labels in loader:
                # Move to GPU if available
                clips = clips.to(self.device)  
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(clips)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Average loss per epoch
            avg_loss = total_loss / len(loader)

            if epoch % 10 == 0:
                print('-'*50)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
                print(f"Average Loss: {total_loss / len(loader):.4f}")
                print('-'*50)
        
        print('-'*50)
        print(f"Epoch {epochs}, Loss: {avg_loss:.4f}")
        print(f"Average Loss: {total_loss / len(loader):.4f}")
        print('-'*50)

    def evaluate(self):
        '''
        Function to run inference on the trained CV model

        Inputs:
            - model: The trained CV model
            - loader: The test loader to be used for inference
            - device: The gpu/cpu to be used for running inference
        
        Returns:
            - accuracy: the accuracy of the model on the test set
            - recall: the recall score calculated on the test set
        '''
        # move model to gpu and set to evaluation mode
        model = self.model
        loader = self.test_loader

        model.to(self.device) 
        model.eval()

        # create components to calculate metrics
        correct, total = 0, 0
        all_labels, all_predictions = [], []

        # run inference over the test data loader
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Running inference"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # calculate and return the accuracy and recall over the test set
        recall = recall_score(all_labels, all_predictions, average='macro')
        return correct / total, recall

    def get_clip_frames(self, frames, time, clip_size, fps, time_splits):
        '''
        Get only the frames for the given time bin

        Inputs:
            - frames: the entire list of video frames
            - time: the current time stamp which is an end time
            - clip_size: the number of frames to return
            - fps: the fps of the video
            - time_splits: the frequency for which times were binned
        
        Returns:
            - A tensor of the necessary clip frames
        '''
        # get the start time of the video
        start_time_in_seconds = max(0, time - time_splits)

        # get the start frame
        start_frame = int(start_time_in_seconds * fps)

        # get the correct frames to be returned
        clip_frames = frames[start_frame:start_frame + clip_size]
        return torch.stack(clip_frames, dim=1)


    def add_frames_to_df(self, df):
        '''
        Given a dataframe, create a dictionary connecting rows to video frames
        
        Inputs:
            - df: The dataframe connecting users to videos and gaze data
        
        Returns:
            - df: The dataframe with new columns linking rows to the dictionary
            - frames_dict: the dictionary used for training
        '''
        df = pd.read_csv(self.binned_video_csv, index_col=0)
        frames_dict = {}
        df = df.assign(
            clip_frames_id=None,
            fps=None
        )

        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        frames_per_video = 36*15
        time_splits = df.iloc[1]['time']
        frames_per_clip = int(36*time_splits)

        for video_id, group in df.groupby('videoId'):
            print(f"Processing videoId: {video_id}")
            cap = cv2.VideoCapture(os.path.join(self.videos_dir, f'{video_id}.mp4'))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Load existing video and grab frames
            frames_with_positions = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames_with_positions.append(frame)

            frames = [transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames_with_positions]
            while len(frames) < frames_per_video:
                frames.append(frames[-1]) 

            for idx, row in group.iterrows():
                clip_frames = self.get_clip_frames(frames=frames, time=row['time'], clip_size=frames_per_clip, fps=fps, time_splits=time_splits)

                frames_dict[idx] = clip_frames  # Store tensor in dictionary
                df.at[idx, 'clip_frames_id'] = idx  # Store reference
                df.at[idx, 'fps'] = fps
            
            cap.release()
            cv2.destroyAllWindows()
        
        return df, frames_dict

    def run_training(self):
        '''
        Sets up the training and test loaders, and runs training

        Inputs:
            - df: the dataframe used for training
            - frames_dict: the dictionary that contains the video frames
            - device: the gpu/cpu used for training
        
        Returns:
            - model: the trained model
            - train_loader: the training set
            - test_loader: the test set
        '''
        df = self.df
        frames_dict = self.frames_dict
        training_df = df.drop(columns=['hazard'])
        labels = df['hazard']

        X_train, X_test, y_train, y_test = train_test_split(training_df, labels, test_size=0.1, stratify=labels, random_state=42)

        frames_dict_train = {idx: frames_dict[row['clip_frames_id']] for idx, row in X_train.iterrows()}
        frames_dict_test = {idx: frames_dict[row['clip_frames_id']] for idx, row in X_test.iterrows()}
        
        train_dataset = VideoDataset(df=X_train, labels=y_train, frames_dict=frames_dict_train)
        test_dataset = VideoDataset(df=X_test, labels=y_test, frames_dict=frames_dict_test)

        sample_weights = self.get_class_weights_from_labels(y_train).cpu().numpy()

        # WeightedRandomSampler for oversampling
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

        # Load the pre-trained I3D model
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        for param in model.parameters():
            param.requires_grad = False

        # Modify the final layer for binary classification
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

        self.training_loop(model, train_loader, epochs=1000)

        self.model = model
        self.test_loader = test_loader
        
        print(f"\nAttempting to save the DEEP LEARNING MODEL...")
        torch.save(self.model, self.model_checkpoint)

        print(f"✅ Model saved successfully: {self.model_checkpoint}")

        return model, train_loader, test_loader
    
    def predict_one(
        self,
        videoId: str,
    ):
        
        
        device = self.device
        
        # Load the trained model from checkpoint.
        model_path = self.model_checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        model.to(device)
        
        # Load the CSV file that was used during training.
        csv_path = self.binned_video_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Binned CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Filter the CSV for the given videoId.
        df_video = df[df['videoId'] == videoId]
        if df_video.empty:
            raise ValueError(f"No entries found for videoId {videoId} in {csv_path}")
        
        # For determining clip parameters, we mimic the training code:
        # In your training, you use:
        #   time_splits = df.iloc[1]['time']
        #   frames_per_clip = int(36 * time_splits)
        # Here we do the same (using the second row if available).
        if len(df_video) > 1:
            time_splits = df_video.iloc[1]['time']
        else:
            time_splits = df_video.iloc[0]['time']
        frames_per_clip = int(36 * time_splits)
        # Also define the expected number of frames per video (as in your training code)
        frames_per_video = 36 * 15

        # Load the video file.
        video_path = os.path.join(self.videos_dir, f"{videoId}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        # Get video FPS (frames per second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Define the image transform (same as used during training).
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Read and transform all video frames.
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert from BGR (OpenCV) to RGB, then to a PIL Image, then apply transform.
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor_frame = transform(image)
            frames.append(tensor_frame)
        cap.release()
        cv2.destroyAllWindows()
        
        # If the video has fewer frames than expected, pad with the last frame.
        if len(frames) < frames_per_video:
            frames.extend([frames[-1]] * (frames_per_video - len(frames)))
        
        # Helper function to extract clip frames for a given time stamp.
        def get_clip_frames(frames, time_val, clip_size, fps, time_splits):
            """
            Extracts a clip given a time stamp (the end time of the clip) by computing
            the start frame using the provided time_splits value.
            """
            start_time_in_seconds = max(0, time_val - time_splits)
            start_frame = int(start_time_in_seconds * fps)
            clip_frames = frames[start_frame : start_frame + clip_size]
            # In case there are not enough frames at the end, pad with the last frame.
            if len(clip_frames) < clip_size:
                clip_frames.extend([clip_frames[-1]] * (clip_size - len(clip_frames)))
            # Stack the list of tensors along the temporal dimension.
            return torch.stack(clip_frames, dim=1)  # Shape: [3, clip_size, 112, 112]
        
        # List to store prediction results.
        results = []

        # For each time bin (row) in the filtered CSV, extract the clip and run prediction.
        for idx, row in df_video.iterrows():
            time_val = row['time']  # This is used to define the clip
            # Extract the clip frames (tensor shape will be [3, frames_per_clip, 112,112])
            clip_tensor = get_clip_frames(frames, time_val, frames_per_clip, fps, time_splits)
            # Add a batch dimension: shape becomes [1, 3, frames_per_clip, 112,112]
            clip_tensor = clip_tensor.unsqueeze(0).to(device)
            
            # Run the model on this clip.
            with torch.no_grad():
                output = model(clip_tensor)
                # Assuming the model outputs logits for 2 classes,
                # compute softmax probabilities and pick the predicted class.
                probs = torch.softmax(output, dim=1)
                predicted_label = torch.argmax(probs, dim=1).item()
                probability = probs[0, predicted_label].item()
            
            # Append the result (you can include additional fields if needed).
            results.append({
                "videoId": videoId,
                "time": time_val,
                "predicted_label": predicted_label,
                "probability": probability
            })
        
        # Convert the results list to a DataFrame and write to CSV.
        results_df = pd.DataFrame(results)
        output_csv_path = os.path.join(self.output_dir, f'{videoId}.csv')
        results_df.to_csv(output_csv_path, index=False)
        print(f"Prediction results saved to {output_csv_path}")
        
        return results_df
    
    
            
def main():
    print()


    # model, _, test_loader = run_training(df, frames_dict, device)

    # accuracy, recall = evaluate(model, test_loader, device)
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")
    # print(f"Test Recall: {recall * 100:.2f}%")  

if __name__ == '__main__':
    main()