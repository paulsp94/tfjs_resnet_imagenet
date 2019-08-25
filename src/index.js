import * as tf from '@tensorflow/tfjs-node';
import Jimp from 'jimp';
import labels from '../assets/labels.json';

export default class ResNetPrediction {
  constructor() {
    this.model;
    this.labels = labels;
  }

  async initialize() {
    this.model = await tf.loadLayersModel(
      `file:///${__dirname}/../ResNet50/model.json`
    );
  }

  static async create() {
    const o = new ResNetPrediction();
    await o.initialize();
    return o;
  }

  loadImg = async imgURI => {
    return Jimp.read(imgURI).then(img => {
      img.resize(224, 224);
      const p = [];
      img.scan(0, 0, img.bitmap.width, img.bitmap.height, function test(
        x,
        y,
        idx
      ) {
        p.push(this.bitmap.data[idx + 0]);
        p.push(this.bitmap.data[idx + 1]);
        p.push(this.bitmap.data[idx + 2]);
      });

      return tf
        .tensor3d(p, [img.bitmap.width, img.bitmap.height, 3])
        .reshape([1, img.bitmap.width, img.bitmap.height, 3]);
    });
  };

  classify = async imgURI => {
    const img = await this.loadImg(imgURI);
    const predictions = await this.model.predict(img);
    const prediction = predictions
      .reshape([1000])
      .argMax()
      .dataSync()[0];
    const result = this.labels[prediction];
    return result;
  };
}
