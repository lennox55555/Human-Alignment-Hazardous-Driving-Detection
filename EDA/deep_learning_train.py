import cv2
import pandas as pd
import numpy as np
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

def get_class_weights(loader, device):
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
    return torch.FloatTensor(class_weights).to(device)

def get_class_weights_from_labels(labels, device):
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
    return torch.FloatTensor(class_weights).to(device)
    
def training_loop(model, loader, device, epochs=10):
    '''
    The training loop for the model

    Inputs: 
        - model: the 3D computer vision model used for training
        - loader: the train loader
        - device: the gpu/cpu being used for trainng
        - epochs: the number of epochs used for training
    '''
    # Class Weights
    class_weights = get_class_weights(loader, device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Move model to device and set to training mode
    model.to(device)
    model.train() 

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0

        for clips, labels in loader:
            # Move to GPU if available
            clips = clips.to(device)  
            labels = labels.to(device)

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

def evaluate(model, loader, device):
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
    model.to(device) 
    model.eval()

    # create components to calculate metrics
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    # run inference over the test data loader
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Running inference"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # calculate and return the accuracy and recall over the test set
    recall = recall_score(all_labels, all_predictions, average='macro')
    return correct / total, recall

def get_clip_frames(frames, time, clip_size, fps, time_splits):
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

def add_frames_to_df(df):
    '''
    Given a dataframe, create a dictionary connecting rows to video frames
    
    Inputs:
        - df: The dataframe connecting users to videos and gaze data
    
    Returns:
        - df: The dataframe with new columns linking rows to the dictionary
        - frames_dict: the dictionary used for training
    '''
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
    time_splits = df.loc[1, 'time']
    frames_per_clip = int(36*time_splits)

    for video_id, group in df.groupby('videoId'):
        print(f"Processing videoId: {video_id}")
        cap = cv2.VideoCapture(f'./data/training_videos/{video_id}.mp4')
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
            clip_frames = get_clip_frames(frames=frames, time=row['time'], clip_size=frames_per_clip, fps=fps, time_splits=time_splits)

            frames_dict[idx] = clip_frames  # Store tensor in dictionary
            df.at[idx, 'clip_frames_id'] = idx  # Store reference
            df.at[idx, 'fps'] = fps
        
        cap.release()
        cv2.destroyAllWindows()
    
    return df, frames_dict

def run_training(df, frames_dict, device):
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
    training_df = df.drop(columns=['hazard'])
    labels = df['hazard']

    X_train, X_test, y_train, y_test = train_test_split(training_df, labels, test_size=0.1, stratify=labels, random_state=42)

    frames_dict_train = {idx: frames_dict[row['clip_frames_id']] for idx, row in X_train.iterrows()}
    frames_dict_test = {idx: frames_dict[row['clip_frames_id']] for idx, row in X_test.iterrows()}
    
    train_dataset = VideoDataset(df=X_train, labels=y_train, frames_dict=frames_dict_train)
    test_dataset = VideoDataset(df=X_test, labels=y_test, frames_dict=frames_dict_test)

    sample_weights = get_class_weights_from_labels(y_train, device).cpu().numpy()

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

    training_loop(model, train_loader, device, epochs=100)

    return model, train_loader, test_loader
            
def main():
    df = pd.read_csv('./binned_video_dat_wo_user.csv', index_col=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df, frames_dict = add_frames_to_df(df)

    model, _, test_loader = run_training(df, frames_dict, device)

    accuracy, recall = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Recall: {recall * 100:.2f}%")  

if __name__ == '__main__':
    main()