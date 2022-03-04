# Final Project

This is a **group assignment**.

## Milestone 2 - Code Implementation & Technical Report

This milestone is to be delivered at the end of the semester, Friday December 10 @ 11:59 PM. Find the complete [rubric](https://ufl.instructure.com/courses/435230/assignments/4907188) in the Canvas assignment.

## Training Data

The training data set is the same for every team in this course.

You can download the data in our Canvas page:
* ["data_train.npy"](https://ufl.instructure.com/files/63371914/download?download_frd=1)
* ["labels_train.npy"](https://ufl.instructure.com/files/63371448/download?download_frd=1)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas">
    <img src="1-3-3.jpg" alt="Logo" width="200" height="200">
  </a>

  <h3 align="center">README for Atlas Project</h3>

  <p align="center">
    Joel Alvarez, Lianan Armil, Raul Rolon, Christian Walk
    <br />
    <a href="https://github.com/catiaspsilva/README-template/blob/main/images/docs.txt"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://drive.google.com/file/d/1jEKEDa3zINeTxvFW4Si39AeQ_CwYGS0Q/view?usp=sharing">Download Model</a>
    ·
    <a href="https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas/issues">Report Bug</a>
    ·
    <a href="https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#thank you">Thank You</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

We implemented several versions of convolutional neural networks to classify images into 10 classes. Several dimensionality reduction techniques were tested including PCA and converting from RGB to grayscale. We also investigated the effect of changing parameters such as number of epochs, learning rate, regularization, batch size, and kernel size. The models built using transfer learning achieved validation accuracy scores above 90%, while the models built from scratch were only able to achieve 10% accuracy in the validation set. The transfer learning models were built using the [ResNet50V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/ResNet50V2) and [Xception](https://keras.io/api/applications/xception/) structures, and the links to these models can be found below. The most accurate model was constructed by unfreezing the last 10 layers of Xception, which yielded a 95% validation accuracy.

See the [project report](https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas/blob/main/EEE_4773_Final_Project.pdf) for more information

<figure>
  <img src="xception_conf_mat.png" width="400">
  <figcaption align="center"> Confusion matrix for Xception model with 10 layers unfrozen </figcaption>
</figure>

<!-- GETTING STARTED -->
## Getting Started

[Download the Model!](https://drive.google.com/file/d/1jEKEDa3zINeTxvFW4Si39AeQ_CwYGS0Q/view?usp=sharing)

After downloading the model, add it to the same directory as [test.ipynb](https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas/blob/main/test.ipynb). Change YOUR-DATA.npy in `data_test = np.load('YOUR-DATA.npy')` to your data (make sure to put it in the directory or link the file path) and change YOUR-DATA.npy in `labels_test = np.load('YOUR-DATA.npy')` to your labels.

<!-- DEPENDENCIES -->
### Dependencies

* NumPy 1.19.2
  ```sh
  conda install -c conda-forge numpy
  ```  
* Matplotlib 3.3.2
  ```sh
  conda install -c conda-forge matplotlib  
  ```  
* Tensorflow 2.7.0
  ```sh
  conda install -c conda-forge tensorflow
  ```
* SciKit Learn 0.23.2
  ```sh
  conda install -c conda-forge scikit-learn
  ```
* Seaborn 0.11.0
  ```sh
  conda install -c conda-forge seaborn
  ```
* Pandas 1.1.3
  ```sh
  conda install -c conda-forge pandas
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas.git
   ```
2. Setup your environment with the same versions of dependencies

<!-- USAGE EXAMPLES -->
## Usage

Here is an example of how the input data and labels are passed into the test function.
<img src="demo.png" width="550">

Once the data is loaded, run the rest of the cells from top to bottom.

_For more examples, please refer to the [Documentation](https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas/blob/main/EEE_4773_Final_Project.pdf)_

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas/issues) for a list of proposed features (and known issues).

<!-- Authors -->
## Authors

Joel Alvarez - joelalvarez@ufl.edu  
Lianan Armil - larmil@ufl.edu  
Raul Rolon - rolonr@ufl.edu  
Christian Walk - walk.christian@ufl.edu

Project Link: [https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas](https://github.com/Fundamentals-of-Machine-Learning-F21/final-project---code-report-atlas)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

You can acknowledge any individual, group, institution or service.
* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)

<!-- thank you -->
## Thank you

Thank you Dr. Silva for a wonderful semester!
