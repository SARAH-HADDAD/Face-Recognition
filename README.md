# Face Recognition based on MTCNN and Facenet

This project aims to develop a face recognition application using the MTCNN and Facenet libraries. MTCNN is used to detect faces in images, while Facenet is used to encode the detected faces into a unique vector that can be compared to other vectors for performing face recognition.

## Prerequisites

Before you can run this project, you must have the following packages installed:
```
tensorflow
mtcnn
keras
opencv
scikit-learn
```
## Execution

To run this project, you can clone this repository to your local machine using the following command:
```
git clone https://github.com/verus56/face-recognition-based-sur-mtcnn-facenet.git
```
Once cloned, you can run the main file main.py using the following command:
```
python main.py
```
The application will start and you will be able to start using the Facial recognition system by adding images of people to the database.

## Internal functioning

When the application starts, it uses MTCNN to detect faces. The detected faces are then encoded using Facenet to produce a unique vector that represents each person.

When an image is submitted for face recognition, the same face detection and encoding process is performed on the submitted image. The resulting vector is then compared to each vector in the face recognition database to find the closest face.

## Conclusion

This project shows you how to use the MTCNN and Facenet libraries to develop a simple face recognition application. You can use this code as a base to build a more complex application or adapt it to your needs.

## Resources

#### Report: 
https://docs.google.com/document/d/1qUyRCTYTMTFNUmzSWFhkliT4pCzB6jjhc4pN74gVn50/edit?usp=sharing
#### Presentation: 
https://www.canva.com/design/DAFVO0BOhDQ/-aJzte6I3DqJtJX14fWbyw/view?utm_content=DAFVO0BOhDQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

#### by: 
```
  1. HAMZAOUI Thameur
  2. OMARI Hamza
  3. HADDAD SARAH
  4. ELFECIH SARAH
```
