import os
from shutil import copyfile

def creating_dataset():
    # Set the paths based on the parent directory of Standford40
    images_path = "./Standford40/JPEGImages"
    labels_path = "./Standford40/ImageSplits"
    new_dataset_path = "./StandfordActionDataset"

    # Check if the necessary directories exist, if not create them
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
        os.mkdir(os.path.join(new_dataset_path, 'train'))
        os.mkdir(os.path.join(new_dataset_path, 'test'))

    # Ensure 'actions.txt' exists in the labels folder
    actions_file = os.path.join(labels_path, 'actions.txt')
    if not os.path.exists(actions_file):
        print(f"Error: {actions_file} not found.")
        return

    # Read the actions from the actions.txt file, cleaning up extra data
    with open(actions_file, 'r') as f:
        actions = []
        for line in f.readlines():
            line = line.strip()  # Remove any leading/trailing spaces
            if line:
                # Extract the action name before the first tab or space
                action = line.split('\t')[0]
                actions.append(action)

    # Process train.txt and test.txt
    for file_name in ['train.txt', 'test.txt']:
        file_path = os.path.join(labels_path, file_name)
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            continue

        with open(file_path, 'r') as f:
            txt_contents = f.read()
        image_names = txt_contents.split('\n')

        # Determine whether we're processing 'train' or 'test'
        train_or_test = file_name.split('.')[0]

        # Process each image in the file
        for image_name in image_names[:-1]:  # Skip the last empty line
            # Extract the full action name from the image filename (e.g., 'washing_dishes_178.jpg' -> 'washing_dishes')
            action = image_name.split('_')[0]  # Extract the first part of the filename
            
            # Rebuild the full action name by including all parts before the number (e.g., 'washing_dishes')
            if len(image_name.split('_')) > 2:
                action = '_'.join(image_name.split('_')[:-1])  # Rejoin all parts before the number
            
            # Debugging: Print the action extracted from the image filename and the valid actions for comparison
            print(f"Extracted action: {action}")
            print(f"Valid actions: {actions}")
            
            # Check if the action is in the list of actions
            if action not in actions:
                print(f"Skipping image {image_name} due to unrecognized action: {action}")
                continue

            # Create directories for the action if they don't exist
            action_folder = os.path.join(new_dataset_path, train_or_test, action)
            if not os.path.exists(action_folder):
                os.makedirs(action_folder)

            # Source image path
            image_path = os.path.join(images_path, image_name)

            # Ensure the image exists before copying
            if os.path.exists(image_path):
                copyfile(image_path, os.path.join(action_folder, image_name))
            else:
                print(f"Image {image_name} not found in {images_path}")

    print("Image sorting and dataset creation complete.")

# Run the function to create the dataset
creating_dataset()
