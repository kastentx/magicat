require('dotenv').config()
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const fs = require('fs')
const argv = require('yargs').argv
const sharp = require('sharp')
const { createCanvas, Image } = require('canvas')
const canvas = createCanvas(513, 513)
const ctx = canvas.getContext('2d')
const filename = argv._[0]
let processImg
let MAXData

const MODEL_PATH = 'file://tensorflowjs_model.pb'
const WEIGHTS_PATH = 'file://weights_manifest.json'

const OBJ_LIST = ['background', 'airplane', 'bicycle', 'bird', 'boat', 
'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 
'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 
'sofa', 'train', 'tv']
let objMap = {} 
OBJ_LIST.forEach((x,i)=> objMap[x]=i)
const OBJ_MAP = objMap

const URLtoB64 = dataURL => dataURL.split(',')[1]

const COLOR_MAP = {
  green: [0, 128, 0],
  red: [255, 0, 0],
  blue: [0, 0, 255],
  purple: [160, 32, 240],
  pink: [255, 185, 80],
  teal: [0, 128, 128],
  yellow: [255, 255, 0],
  gray: [192, 192, 192]
}

const COLOR_LIST = Object.values(COLOR_MAP)

const getColor = pixel => COLOR_LIST[pixel - 1]

const cleanTFJSResponse = (modelOutput) => {
  const objIDs = [...new Set(modelOutput)] // eslint-disable-next-line
  const objPixels = modelOutput.reduce((a, b) => (a[OBJ_LIST[b]] = ++a[OBJ_LIST[b]] || 1, a), {})
  const objTypes = objIDs.map(x => OBJ_LIST[x])
  return {
    foundSegments: objTypes.concat('colormap'),
    response: {
      objectTypes: objTypes,
      objectIDs: objIDs,
      objectPixels: objPixels,
      flatSegMap: modelOutput
    }
  }
}

const invisibleSegment = (segmentName, imageObj, data) => {
  return new Promise((resolve, reject) => {
    let img = new Image()
    let imageURL
    img.onload = () => {
      try {
        const flatSegMap = imageObj.response.flatSegMap
        ctx.drawImage(img, 0, 0, img.width, img.height)
        const imageData = ctx.getImageData(0, 0, img.width, img.height)
        const data = imageData.data
        if (segmentName === 'colormap') {
          for (let i = 0; i < data.length; i += 4) {
            const segMapPixel = flatSegMap[i / 4]
            let objColor = [0, 0, 0]
            if (segMapPixel) {
              objColor = getColor(imageObj.response.objectIDs.indexOf(segMapPixel))
              data[i]   = objColor[0]  // red channel
              data[i+1] = objColor[1]  // green channel
              data[i+2] = objColor[2]  // blue channel
              data[i+3] = 200          // alpha
            }
          }
        } else { 
          for (let i = 0; i < data.length; i += 4) {
            const segMapPixel = flatSegMap[i / 4]
            if (segMapPixel !== OBJ_MAP[segmentName]) {
              data[i+3] = 0           // alpha
            }
          }
        }
        ctx.putImageData(imageData, 0, 0)      
        imageURL = canvas.toDataURL()
        resolve(URLtoB64(imageURL))
      } catch (e) {
        reject(`${e} - image load error`)
      }
    }
    img.src = data
  })
}

const saveSegment = async (filename, segName, MAXData, data) => {
  const outputName = `${filename.split('.')[0]}-${ segName }.png`
  console.log(`saved ${ outputName }`)
  fs.writeFileSync(outputName, Buffer.from(await invisibleSegment(segName, MAXData, data), 'base64'))
  return null
}

if (filename) { 
  sharp(filename)
    .resize(513, 513, {
      fit: 'inside'
    }).toBuffer()
    .then(async data => {
      try {
        const img = new Image()
        img.onload = async () => await ctx.drawImage(img, 0, 0)
        img.onerror = err => { throw err }
        img.src = data
        const myTensor = tf.fromPixels(canvas).expandDims()   
        const model = await tf.loadFrozenModel(MODEL_PATH, WEIGHTS_PATH)
        MAXData = cleanTFJSResponse(Array.from(model.predict(myTensor).dataSync()))
        console.log(`The image '${ filename }' contains the following segments: ${ MAXData.response.objectTypes.join(', ') }.`)
      } catch (e) {
        console.error(`error processing image - ${ e }`)
        return null
      } finally {
        if (argv.save) {
          if (argv.save === 'all') {
            MAXData.foundSegments.forEach(seg => {
              saveSegment(filename, seg, MAXData, data)
            })
          } else if (argv.save !== true) {
            saveSegment(filename, argv.save, MAXData, data)
          } else {
            console.log(`\nAfter the --save flag, provide an object name from the list above, or 'all' to save each segment individually.`)
          }
        }
      }
    })
} else {
  console.error(`no input image specified.`)
}