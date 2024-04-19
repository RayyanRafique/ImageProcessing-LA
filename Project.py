from pickle import NONE
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#Face recognition
print("Task 1 Face Recognition\n\n")
def load_images(path, new_width=50, new_height=50): 
    images = []
    labels = []
    for subject_folder in os.listdir(path):
        subject_path = os.path.join(path, subject_folder)
        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                if filename.endswith(".png"):
                    img_path = os.path.join(subject_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, (new_width, new_height))
                    images.append(img_resized.flatten())  # Flatten the resized image to a 1D array
                    labels.append(subject_folder)
    return np.array(images), np.array(labels) 

def split_dataset(images, labels, num_persons=15, num_images_per_person=11, num_train_images=10):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for person_id in range(num_persons):
        start_index = person_id * num_images_per_person 
        end_index = start_index + num_train_images 
        train_images.extend(images[start_index:end_index]) 
        train_labels.extend(labels[start_index:end_index]) 
        test_images.extend(images[end_index:end_index + 1]) 
        test_labels.extend(labels[end_index:end_index + 1]) 
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def perform_pca(train_images, num_components=0.99):
    mean_face = np.mean(train_images, axis=0)
    centered_images = train_images - mean_face
    cov_matrix = np.cov(centered_images, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components_to_keep = np.argmax(cumulative_energy >= num_components) + 1
    pca_basis = eigenvectors[:, :num_components_to_keep]
    return mean_face, pca_basis

def project_images(test_images, mean_face, pca_basis):
    centered_images = test_images - mean_face
    projections = np.dot(centered_images, pca_basis)
    return projections

def back_project(projections, mean_face, pca_basis):
    reconstructed_images = np.dot(projections, pca_basis.T) + mean_face
    return reconstructed_images

def calculate_loss(original_images, reconstructed_images):
    loss = np.mean(np.square(original_images - reconstructed_images))
    return loss

def load_Test_Image(image_path, new_width=50, new_height=50):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (new_width, new_height))
    return img_resized.flatten()



def face_Recognition_SingleImage(dataset_path, test_image_path, num_components=0.99):
    images, labels = load_images(dataset_path)
    train_images, train_labels, _, _ = split_dataset(images, labels)
    chosen_threshold = 0.1 * np.max([calculate_loss(img1, img2) for img1 in train_images for img2 in train_images])
    test_image = load_Test_Image(test_image_path)
    mean_face, pca_basis = perform_pca(train_images, num_components)
    test_projection = np.dot(test_image - mean_face, pca_basis)
    reconstructed_image = np.dot(test_projection, pca_basis.T) + mean_face
    loss = calculate_loss(test_image, reconstructed_image)
    predicted_label = "Unknown" 
    min_loss_index = np.argmin([calculate_loss(train_img, reconstructed_image) for train_img in train_images])
    predicted_label = train_labels[min_loss_index]

    #display the images
    plt.figure(figsize=(8, 4))

    #original Image output
    plt.subplot(1, 3, 1)
    plt.imshow(test_image.reshape((50, 50)), cmap='gray')
    plt.title("Original Image")
    
    #reconstructed Image output
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_image.reshape((50, 50)), cmap='gray')
    plt.title("Reconstructed Image")
    
    #display the recognized label as output
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f"Predicted Label: {predicted_label}", fontsize=12, ha='center', va='center')
    plt.axis('off')
    
    plt.show()

    return predicted_label

dataset_path = "D:/BSCS22122-LA-Project/salefaces"
test_image_path = "D:/BSCS22122-LA-Project/salefaces/subject11/subject11.sad.png"
predicted_label = face_Recognition_SingleImage(dataset_path, test_image_path, num_components=0.99)
print("Predicted Label:", predicted_label)



print("\nTask 2 Expression Recognition!!\n\n")

def load_ImagesExpression(path, expressions, num_persons=15, new_width=50, new_height=50):
    images = {expr: [] for expr in expressions}
    labels = {expr: [] for expr in expressions}
    num_images_per_person = 15
    for expr in expressions:
        expr_folder = os.path.join(path, expr)
        if os.path.isdir(expr_folder):
            print(f"Processing expression folder: {expr}")
            for person_id in range(1, num_persons + 1):
                for img_number in range(1, num_images_per_person + 1):
                    filename = f"subject{person_id:02d}.{expr}.png"
                    img_path = os.path.join(expr_folder, filename)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # Resize the image
                            img_resized = cv2.resize(img, (new_width, new_height))
                            images[expr].append(img_resized.flatten())
                            labels[expr].append(person_id)
                        else:
                            print(f"    Error loading image: {img_path}")
                    except Exception as e:
                        print(f"    Error processing image: {img_path}")
                        print(e)
        else:
            print(f"Expression folder not found: {expr_folder}")
    return images, labels


def split_ExpressionDataset(images, labels, expressions, num_train_persons=10):
    train_images = {expr: [] for expr in expressions}
    train_labels = {expr: [] for expr in expressions}
    test_images = {expr: [] for expr in expressions}
    test_labels = {expr: [] for expr in expressions}
    num_persons = 15
    num_images_per_person = 1
    for person_id in range(num_persons):
        for expr in expressions:
            expr_images = images[expr][person_id * num_images_per_person: (person_id + 1) * num_images_per_person]
            if person_id < num_train_persons:
                train_images[expr].extend(expr_images)
                train_labels[expr].extend([person_id] * len(expr_images))
            else:
                test_images[expr].extend(expr_images)
                test_labels[expr].extend([person_id] * len(expr_images))

    return train_images, train_labels, test_images, test_labels

# Function to perform PCA on facial expression training images
def perform_ExpressionPCA(train_images, expr, num_components=0.99):
    mean_face = np.mean(train_images[expr], axis=0)
    centered_images = train_images[expr] - mean_face
    cov_matrix = np.cov(centered_images, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components_to_keep = np.argmax(cumulative_energy >= num_components) + 1
    pca_basis = eigenvectors[:, :num_components_to_keep]
    return mean_face, pca_basis

# Function to project test images onto PCA basis and perform expression recognition
def recognize_Expression(test_images, expressions, pca_bases, mean_faces):
    predictions = []
    for expr in expressions:
        expr_predictions = []
        for test_img in test_images[expr]:
            min_loss = float('inf')
            predicted_expr = "Unknown"
            for train_img, pca_basis, mean_face in zip(train_images[expr], pca_bases[expr], mean_faces[expr]):
                centered_test_img = test_img - mean_face
                test_projection = np.dot(centered_test_img, pca_basis)
                reconstructed_img = np.dot(test_projection, pca_basis.T) + mean_face
                loss = np.mean(np.square(test_img - reconstructed_img))
                if loss < min_loss:
                    min_loss = loss
                    predicted_expr = expr
            expr_predictions.append(predicted_expr)
        predictions.append(expr_predictions)
    return predictions


# Function to load a single image for testing
def load_Test_Image(image_path, new_width=50, new_height=50):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (new_width, new_height))
            return img_resized.flatten()
        else:
            print(f"Error loading image: {image_path}")
            return None
    except Exception as e:
        print(f"Error processing image: {image_path}")
        print(e)
        return None

# Function to test a single image for facial expression recognition
def test_SingleImage(image_path, expressions, pca_bases, mean_faces):
    test_image = load_Test_Image(image_path)
    if test_image is not None:
        predictions = recognize_Expression({expr: [test_image] for expr in expressions}, expressions, pca_bases, mean_faces)
        return predictions
    else:
        return None


dataset_path = "D:/BSCS22122-LA-Project/salefaces/expressions"
expressions = ["happy", "normal", "sad", "sleepy", "surprised", "wink"]
num_train_persons = 10

images, labels = load_ImagesExpression(dataset_path, expressions)
train_images, train_labels, test_images, test_labels = split_ExpressionDataset(images, labels, expressions, num_train_persons)

pca_bases = {expr: [] for expr in expressions}
mean_faces = {expr: [] for expr in expressions}

for expr in expressions:
    mean_face, pca_basis = perform_ExpressionPCA(train_images, expr, num_components=0.99)
    pca_bases[expr].append(pca_basis)
    mean_faces[expr].append(mean_face)


def display_SingleExpression(image, expression):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.reshape((50, 50)), cmap='gray')
    plt.title(f"Expression: {expression}")
    plt.axis('off')
    plt.show()

test_image_path = "D:/BSCS22122-LA-Project/salefaces/expressions/happy/subject01.happy.png"
result = test_SingleImage(test_image_path, expressions, pca_bases, mean_faces)

if result is not None:  
    predicted_expression = result[0][0]
    print(f"Predicted expression for {test_image_path}: {predicted_expression}")
    display_SingleExpression(load_Test_Image(test_image_path), predicted_expression)
else:
    print("Test failed.")



print("Task 3 Glasses/No Glasses recognition\n\n")
    
def load_ImagesFolder(folder_path, num_images, filename_pattern="subject{:02d}.png"):
    images = []
    for image_id in range(1, num_images + 1):
        img_path = os.path.join(folder_path, filename_pattern.format(image_id))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (50, 50))
        images.append(img_resized.flatten())

    return np.array(images)

def load_Glassesdata(glasses_path, no_glasses_path, num_persons=10):
    glasses_images = []
    no_glasses_images = []
    for person_id in range(1, num_persons + 1):
        glasses_folder = glasses_path
        glasses_images.extend(load_ImagesFolder(glasses_folder, num_images=1, filename_pattern="subject{:02d}.glasses.png"))
        no_glasses_folder = no_glasses_path
        no_glasses_images.extend(load_ImagesFolder(no_glasses_folder, num_images=1, filename_pattern="subject{:02d}.noglasses.png"))

    return np.array(glasses_images), np.array(no_glasses_images)


def Load_Gimage(path, start_subject=11, end_subject=15, new_width=50, new_height=50): 
    images = []
    labels = []
    i = 0;
    for subject_id in range(start_subject, end_subject + 1):
        subject_folder = f"subject{subject_id:02d}"
        subject_path = os.path.join(path, subject_folder)

        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                if filename.endswith(".png"):
                    img_path = os.path.join(subject_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, (new_width, new_height))
                    images.append(img_resized.flatten())  # Flatten the resized image to a 1D array
                    labels.append(subject_folder) 
                    
    return np.array(images), np.array(labels)

def perform_PCA_Glasses_NoGlasses(glasses_images, no_glasses_images):
    mean_face_glasses, pca_basis_glasses = perform_pca(glasses_images)
    mean_face_no_glasses, pca_basis_no_glasses = perform_pca(no_glasses_images)
    return mean_face_glasses, pca_basis_glasses, mean_face_no_glasses, pca_basis_no_glasses



def test_glassRecognition(test_images, pca_basis_glasses, pca_basis_no_glasses, mean_face_glasses, mean_face_no_glasses):
    im = 1
    for person_id, person_images in enumerate(test_images, start=11):
        if person_images.ndim == 1:
            person_images = np.expand_dims(person_images, axis=0)
        for image_id, test_image in enumerate(person_images, start=1):
            projection_glasses = np.dot(test_image - mean_face_glasses, pca_basis_glasses)
            projection_no_glasses = np.dot(test_image - mean_face_no_glasses, pca_basis_no_glasses)
            reconstructed_glasses = np.dot(projection_glasses, pca_basis_glasses.T) + mean_face_glasses
            reconstructed_no_glasses = np.dot(projection_no_glasses, pca_basis_no_glasses.T) + mean_face_no_glasses
            loss_glasses = np.mean(np.square(test_image - reconstructed_glasses))
            loss_no_glasses = np.mean(np.square(test_image - reconstructed_no_glasses))
            has_glasses = loss_glasses > loss_no_glasses
            predicted_status = "With Glasses" if has_glasses else "Without Glasses"    
            print(f"Test Image{im}: {predicted_status}")   
            im = im+1
            

glasses_path = "D:/BSCS22122-LA-Project/salefaces/glasses"
no_glasses_path = "D:/BSCS22122-LA-Project/salefaces//no glasses"
glasses_images, no_glasses_images = load_Glassesdata(glasses_path, no_glasses_path)

test_glasses_path = "D:/BSCS22122-LA-Project/salefaces/test glasses"
test_images, labels = Load_Gimage(test_glasses_path, start_subject=11, end_subject=15) 
mean_face_glasses, pca_basis_glasses, mean_face_no_glasses, pca_basis_no_glasses = perform_PCA_Glasses_NoGlasses(glasses_images, no_glasses_images)

test_glassRecognition(test_images, pca_basis_glasses, pca_basis_no_glasses, mean_face_glasses, mean_face_no_glasses) 


