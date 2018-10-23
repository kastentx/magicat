require('dotenv').config()
const fs = require('fs')
const { createCanvas, Image, loadImage } = require('canvas')
const canvas = createCanvas(513, 513)
const ctx = canvas.getContext('2d')
const sharp = require('sharp')
const filename = process.argv[2]

const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node-gpu')

const MODEL_PATH = 'file://tensorflowjs_model.pb'
const WEIGHTS_PATH = 'file://weights_manifest.json'

const OBJ_LIST = ['background', 'airplane', 'bicycle', 'bird', 'boat', 
'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 
'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 
'sofa', 'train', 'tv']

if (filename) { 
  sharp(filename)
    .resize(513, 513, {
      fit: 'inside'
    }).toBuffer()
    .then(async data => {
      const img = new Image()
      img.onload = async () => await ctx.drawImage(img, 0, 0)
      img.onerror = err => { throw err }
      img.src = data
      const myTensor = tf.fromPixels(canvas).expandDims()   
      try {
        const model = await tf.loadFrozenModel(MODEL_PATH, WEIGHTS_PATH)
        const modelOutput = Array.from(model.predict(myTensor).dataSync())
        const objIDs = [...new Set(modelOutput)]
        const objTypes = objIDs.map(x => OBJ_LIST[x])
        console.log(`The image '${filename}' contains: ${objTypes.join(', ')}`)
      } catch (e) {
        console.error(`error loading model - ${ e }`)
      }
    }) 
} else {
  console.error(`no input image specified.`)
  return
}