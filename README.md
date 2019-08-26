# ResNet Model trained on ImageNet
[![GSOC](https://img.shields.io/badge/GSOC-2019-yellow)](https://summerofcode.withgoogle.com/organizations/6137730124218368/?sp-page=2#4558376158101504)

The ResNet50 model pretrained on imagenet for TensorFlow.js as a layers model.   
On ImageNet, this model gets to a top-1 validation accuracy of 0.749 and a top-5 validation accuracy of 0.921.   
The default input size for this model is 224x224.   

This model has been converted, using the [tfjs-converter][1].  
The base model and weights were taken from [keras][2].

[1]: https://www.npmjs.com/package/@tensorflow/tfjs-converter
[2]: https://keras.io/applications/#resnet

Install `npm install resnet_imagenet`

## How to use

```javascript
import ResNetPrediction from 'resnet_imagenet';

const catURI = 'https://upload.wikimedia.org/wikipedia/commons/6/66/An_up-close_picture_of_a_curious_male_domestic_shorthair_tabby_cat.jpg';

const run = async () => {
    const predictor = await ResNetPrediction.create();
    const preditction = await predictor.classify();
    return prediciton;
}
```

To try the model you can just load it using:    
```javascript
XceptionURL = 'https://github.com/paulsp94/tfjs_Xception_imagenet/blob/master/model/model.json';
const Xception = await tf.loadLayersModel(XceptionURL);
```