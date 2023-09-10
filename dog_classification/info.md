# Dog Classification Project
<a href="https://github.com/lorenarms/Machine_Learning"><strong> ⇦ Go Back</strong></a>

## The Problem

In this machine learning project, I'll be exploring the feesablity of developing a model that can determine a dog's breed by looking at it's picture. The ultimate goal will be to have a fully trained model that can run predictions on custom images of dogs.

## The Data

The dataset used for this model is supplied by Kagel, the machine learning community website repository containing a variety of datasets and ways to use said sets. This particular set comes from a machine learning contest commissioned by Kagel in which participants develop models that can solve the above mentioned problem. The contest culminates in a submission of the model's prediction probabilities in a csv file format. This project's submission is included in the Git Repository [here](https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/full_model_predictions_submission_1.csv).

The data itself is a set of images of dogs totalling >20,000 jpegs split into two roughly even groups for training and testing. The images are of 120 different dog breeds and the median amount of images for each breed is 82. Below is a distribution of the `train` dataset, in which each picture is paired with a matching breed for training purposes:

![Dog Breed Distribution Graph](https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/breeds.png "Dog Breed Distribution")

## The Model and Environment

The project will make use of Google's Colab Cloud Environment running on a Python 3 Google Computer Engine with a T4 GPU. The Jupyter notebook of the entire project can be viewed [here](https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/Dog_Vision.ipynb).

The model will be based off of the MobileNet V2 130 classification model and employ transfer learning so as to minimize training time.

## Shaping

For the data to work with the selected model it must be shaped appropriately. Images start off as various shapes and sizes, in jpeg format:
<p>
<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/sample_img.jpg" width="400">
</p>
<p>
  Included with the dataset is a csv file of `labels` that contains an array of all 120 unique dog breeds:
</p>

`array(['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller'...]`

<p>
  Since the Keras API needs all data in a computer-friendly format, we will start by converting the images and labels into float and binary datasets, with images linked to their respective labels. Starting with labels, we can first create an array of 120 boolean values, with 'TRUE' for the one label that describes the image correctly, and 'FALSE' for all other labels. An example for the "Boston Bull" is shown below:
</p>

`boston_bull, array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False,  True, False...]`

The above is then converted to an array of binary numbers (0, 1):

`boston_bull, (array([19]),), 19, [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0...]`

Images are converted to Tensors, 3-dimensional arrays of numbers 0 - 255 to represent pixel data:

`array([[[ 89, 137,  89],
        [ 76, 124,  76],
        [ 63, 111,  61],
        ...,`

These arrays are then scaled to floats, represented by values between '0' and '1':

`array([[[0.3264178 , 0.5222886 , 0.3232816 ],
         [0.2537167 , 0.44366494, 0.24117757],
         [0.25699762, 0.4467087 , 0.23893751],
         ...,`

This process is called "normalization". We will take this moment to convert the images to squares before we complete the Tensor creation. Below is a sampling of the images with their respective labels before they are converted to Tensors:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/visualize_data.png" width="400">

## Predictions

Once the model has been fitted with the dataset we can use it to make predictions on the test data. Predictions are returned in a similar format to the converted label array, with each breed represented by a probablity value at an index between 0 - 120. The higher the value, the more confident the model is of it's own prediction. An example is show below:

`[5.02049204e-07 3.98175274e-07 3.70937983e-06 9.97670472e-01
 4.92721085e-07 3.33183743e-06 6.90756281e-07 3.66213203e-06...
...2.60654156e-06 6.02852822e-07 8.18802960e-07 7.72485407e-07
 6.78598795e-08 1.67626399e-06 9.22338862e-04 3.68503497e-05]
Max value (probability of prediction): 0.9976704716682434
Sum: 1.0000001192092896
Max index: 3
Predicted label: airedale`

In the above array, the value at index `3` is shown to be the highest confidence value (`0.9976704716682434`), indicating the model predicts this image to be of an "Airedale".

To better visualize this data we can view it both in graphical form and with a comparison image:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/airdale_correct.png" width="400"> <img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/airdale_image.png" width="400">

The graph above indicates that the option "Airdale" was selected with 99% confidence out of all other breeds, with the "Wire-haired Fox Terrier" the next nearest option to be considered. The image shows an Airdale breed, labeled witht the prediction on the right of the title and the actual value on the left.

This visualization can be helpful in seeing when the model gets a prediction wrong, as well:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/prediction_sample.png" width="400">

In the above image the model predicted that this dog is a "Walker Hound" with 80% confidence. However, this is infact labeled as an English Foxhound.

We can view multiple images at once by passing in a starting index to a custom function:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/train_sample_predictions.png" width="400">





## Results

Not all images are straghtforward enough for th model to discern properly, and reshaping the resolution seems to have some adverse results. Consider the below image of what is labeled as a Keshond but was predicted as a Norwegian Elkhound, along with a comparison photo of a Norwegian Elkhound:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/sample_incorrect_image.png" width="400"> <img src="https://www.k9rl.com/wp-content/uploads/2017/02/Norwegian-Elkhound-dog.jpg" width="400">

From this it is fair to draw the conclusion that some images are better suited for machine learning models than others. The original submission to the model shows distortion from the resolution change, as wel as having another subject taking up nearly 50% of the picture. Overall the model performed well during training, showing an increase in accuaracy on both the training and validation datasets:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/training_epoch_accuracy.png" width="400">

*Blue: training accuracy, Pink: validation accuracy*

Additionally, not all dog breeds are fully represented, and dogs that are considered 'mutts' prove to be a challenge for the model. As mentioned above, the ultimate goal was to train the model to recognize dogs from custom images not found in the original datasets. Below is a sampling of four images passed through the model:

<img src="https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/img/custom_preds.png" width="600">

The image prediction in quadrant 2 is accurate, while the remaining predictions are not. Upper right image shows a standard 'mutt' that derives it's features from a variety of other breeds, but none are Rhodesian Ridgeback. It is noteworthy that this particular dog has been mis-classified as a Rhodesian Ridgeback in the past. It's status as 'mutt' was confirmed through DNA sampling.

The bottom two images show signs of the model having trouble with difficult-to-interpret photos, as the predicted breeds are somewhat close in appearance to each dog shown (based on single features like coloration or face-shape). 

<img src="https://animalsbreeds.com/wp-content/uploads/2014/12/American-Staffordshire-Terrier-7.jpg" width="300">

*American Staffordshire Terrier*

<img src="https://tevrapet.com/wp-content/uploads/2020/12/AdobeStock_219316170-2048x1366.jpeg.webp" width="300">

*Pembrook Corgi*

The prediction for the image in the fourth quandrant seems to be based more off of coloration than any other feature. It should also be noted that the model's confidence level for this prediction was lower than the others. 

## Conclusion

With the model trained and performing as it should by predicting dog breeds with a fairly high level of accuracy based on a training set of just 1000 images, this project can be seen as a successful proof-of-concept. With a larger training set it can be assumed that the model could be trained to predict with an even higher degree of accuracy, and potentially be able to distinguish even similar breeds from one another. For more information about the various custom functions used to show images and data in graphical form, please see the full notebook file [here](https://github.com/lorenarms/Machine_Learning/blob/main/dog_classification/full_model_predictions_submission_1.csv).

## Contact

<p>Check out my <a href="https://www.youtube.com/channel/UCGtp8PRHgPCQHYoSxbMST8A" target="_blank">YouTube channel</a> for more videos about coding projects I've done.</p>
<p>Also, check out my <a href="http://artllj.com" target="_blank">Personal Website</a> for more information about me, and my <a href="https://www.linkedin.com/in/lorenarms95/" target="_blank">LinkedIn</a> to see if I'd be a good fit for your team. </p>
<h3>Thanks for stopping by!</h3>
<img src="https://github.com/lorenarms/SNHU_CS_370_Emerging_Trends_in_CS/blob/main/images/profile.png" alt="[picture of me]" style="width:100px;">
<p>much love
-L
</p>

<a href="https://github.com/lorenarms/Machine_Learning"><strong> ⇦ Go Back</strong></a>
